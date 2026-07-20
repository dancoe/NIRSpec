/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { INITIAL_PROGRAMS } from './data/programs';
import { ProgramReview, CompletionFilter, SortConfig, ColumnConfig, DEFAULT_COLUMNS } from './types';
import { parseSheetDate, recalculateDates } from './utils/dateHelpers';
import { formatPiName } from './utils/nameHelpers';
import ReviewStats from './components/ReviewStats';
import ReviewControls from './components/ReviewControls';
import ReviewTable from './components/ReviewTable';
import EditReviewModal from './components/EditReviewModal';
import AddProgramModal from './components/AddProgramModal';
import { Telescope, RefreshCcw, Info, Calendar, Undo, Redo, Upload, Download, Trash2 } from 'lucide-react';

declare global {
  interface Window {
    electronAPI?: {
      saveFile: (filename: string, content: string) => Promise<{ success: boolean; path?: string; error?: string }>;
      loadBackup: (filename: string) => Promise<{ success: boolean; content?: string; error?: string }>;
    };
  }
}

export default function App() {
  // Constants
  const REFERENCE_DATE = new Date(); // Dynamic today's date
  const month = REFERENCE_DATE.getMonth() + 1;
  const day = REFERENCE_DATE.getDate();
  const yearShort = REFERENCE_DATE.getFullYear().toString().slice(-2);
  const referenceDateString = `${month}/${day}/${yearShort}`;

  // React state with localStorage synchronization and Undo/Redo tracking
  const [programs, setProgramsState] = useState<ProgramReview[]>([]);
  const [past, setPast] = useState<ProgramReview[][]>([]);
  const [future, setFuture] = useState<ProgramReview[][]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [searchTerm, setSearchTerm] = useState('');
  const [completionFilter, setCompletionFilter] = useState<CompletionFilter>('pending');
  const [activeStatFilter, setActiveStatFilter] = useState<string | null>(null);
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'submissionDueDate',
    direction: 'asc',
  });
  
  // Modal states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [addProgramOpen, setAddProgramOpen] = useState(false);
  const [selectedProgram, setSelectedProgram] = useState<ProgramReview | null>(null);

  // Custom dialog / alert state for iframe-safe popups
  const [dialog, setDialog] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    type: 'confirm' | 'alert';
    onConfirm?: () => void;
  } | null>(null);

  const showConfirm = (title: string, message: string, onConfirm: () => void) => {
    setDialog({
      isOpen: true,
      title,
      message,
      type: 'confirm',
      onConfirm,
    });
  };

  const showAlert = (title: string, message: string) => {
    setDialog({
      isOpen: true,
      title,
      message,
      type: 'alert',
    });
  };
  
  const [reviewerFilterEnabled, setReviewerFilterEnabled] = useState(false);
  const [reviewerName, setReviewerName] = useState('Dan Coe');

  const [columns, setColumns] = useState<ColumnConfig[]>(() => {
    const saved = localStorage.getItem('nirspec_table_columns_config_v8');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.length >= DEFAULT_COLUMNS.length) {
          return parsed;
        }
      } catch (err) {}
    }
    return DEFAULT_COLUMNS;
  });

  useEffect(() => {
    localStorage.setItem('nirspec_table_columns_config_v8', JSON.stringify(columns));
  }, [columns]);

  const [rowPitch, setRowPitch] = useState<number>(() => {
    const saved = localStorage.getItem('table_row_pitch_v2');
    return saved ? parseInt(saved, 10) : 0;
  });

  const [colPitch, setColPitch] = useState<number>(() => {
    const saved = localStorage.getItem('table_col_pitch_v2');
    return saved ? parseInt(saved, 10) : 0;
  });

  useEffect(() => {
    localStorage.setItem('table_row_pitch_v2', rowPitch.toString());
  }, [rowPitch]);

  useEffect(() => {
    localStorage.setItem('table_col_pitch_v2', colPitch.toString());
  }, [colPitch]);  // Load programs from Electron backend with localStorage fallback on startup
  useEffect(() => {
    const loadInitialData = async () => {
      let loadedPrograms: ProgramReview[] = [];
      let source = 'localStorage';
      
      if (window.electronAPI) {
        try {
          const res = await window.electronAPI.loadBackup('programs.json');
          if (res.success && res.content) {
            loadedPrograms = JSON.parse(res.content);
            source = 'electron';
          }
        } catch (err) {
          console.error('Failed to load from electron backup:', err);
        }
      }
      
      if (loadedPrograms.length === 0) {
        const saved = localStorage.getItem('nirspec_programs');
        if (saved) {
          try {
            loadedPrograms = JSON.parse(saved);
          } catch (err) {
            console.error('Failed to parse saved programs', err);
          }
        }
      }
      
      const enforced = loadedPrograms.map((p) => {
        const isCycle4 = !p.cycle || p.cycle === '4';
        let updated = { ...p };
        if (isCycle4 && !p.isDeleted) {
          updated.nirspecReviewer = 'Dan Coe';
        }
        if (updated.obsEarliest) {
          const calcs = recalculateDates(updated.obsEarliest);
          if (calcs.reviewDueDate) updated.reviewDueDate = calcs.reviewDueDate;
          if (calcs.submissionDueDate) updated.submissionDueDate = calcs.submissionDueDate;
          if (calcs.finalizeDueDate) updated.finalizeDueDate = calcs.finalizeDueDate;
          if (calcs.flightReadyEarliest) updated.flightReadyEarliest = calcs.flightReadyEarliest;
          if (calcs.flightReadyMid) updated.flightReadyMid = calcs.flightReadyMid;
          if (calcs.flightReadyLatest) updated.flightReadyLatest = calcs.flightReadyLatest;
        }
        return updated;
      });

      setProgramsState(enforced);
      localStorage.setItem('nirspec_programs', JSON.stringify(enforced));
      
      if (window.electronAPI && source !== 'electron') {
        window.electronAPI.saveFile('programs.json', JSON.stringify(enforced, null, 2))
          .catch((err) => console.error('Failed to sync to Electron:', err));
      }
    };
    
    loadInitialData();
  }, []);
  // Centralized program state update that also syncs localStorage and records history
  const saveProgramsToStorage = (updatedList: ProgramReview[], isUndoRedoAction = false, explicitPrevPrograms?: ProgramReview[]) => {
    const enforced = updatedList.map((p) => {
      const isCycle4 = !p.cycle || p.cycle === '4';
      let updated = { ...p };
      if (isCycle4 && !p.isDeleted) {
        updated.nirspecReviewer = 'Dan Coe';
      }
      if (updated.obsEarliest) {
        const calcs = recalculateDates(updated.obsEarliest);
        if (calcs.reviewDueDate) updated.reviewDueDate = calcs.reviewDueDate;
        if (calcs.submissionDueDate) updated.submissionDueDate = calcs.submissionDueDate;
        if (calcs.finalizeDueDate) updated.finalizeDueDate = calcs.finalizeDueDate;
        if (calcs.flightReadyEarliest) updated.flightReadyEarliest = calcs.flightReadyEarliest;
        if (calcs.flightReadyMid) updated.flightReadyMid = calcs.flightReadyMid;
        if (calcs.flightReadyLatest) updated.flightReadyLatest = calcs.flightReadyLatest;
      }
      return updated;
    });

    if (!isUndoRedoAction) {
      const currentSnapshot = explicitPrevPrograms || programs;
      setPast((prevPast) => [...prevPast.slice(-199), currentSnapshot]);
      setFuture([]);
    }

    setProgramsState(enforced);
    localStorage.setItem('nirspec_programs', JSON.stringify(enforced));

    if (window.electronAPI) {
      window.electronAPI.saveFile('programs.json', JSON.stringify(enforced, null, 2))
        .catch((err) => console.error('Failed to save to Electron backup:', err));
    }
  };

  // Undo / Redo actions
  const handleUndo = () => {
    if (past.length === 0) return;
    const previous = past[past.length - 1];
    const newPast = past.slice(0, past.length - 1);

    setPast(newPast);
    setFuture((f) => [programs, ...f]);
    saveProgramsToStorage(previous, true);
  };

  const handleRedo = () => {
    if (future.length === 0) return;
    const nextVal = future[0];
    const newFuture = future.slice(1);

    setFuture(newFuture);
    setPast((p) => [...p, programs]);
    saveProgramsToStorage(nextVal, true);
  };

  // Shortcut key listeners for Undo (Ctrl/Cmd+Z) and Redo (Ctrl/Cmd+Y or Ctrl/Cmd+Shift+Z)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isZ = e.key.toLowerCase() === 'z';
      const isY = e.key.toLowerCase() === 'y';
      const hasMeta = e.metaKey || e.ctrlKey;

      if (hasMeta && isZ) {
        e.preventDefault();
        if (e.shiftKey) {
          handleRedo();
        } else {
          handleUndo();
        }
      } else if (hasMeta && isY) {
        e.preventDefault();
        handleRedo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [past, future, programs]);

  // JSON Save (Download) & Load (Upload) helpers
  const handleSaveJson = () => {
    try {
      const dataStr = JSON.stringify(programs, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
      
      const exportFileDefaultName = `nirspec_mos_apt_programs_review_${new Date().toISOString().slice(0, 10)}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    } catch (err) {
      showAlert('Save Backup Failed', 'Failed to save to JSON file: ' + (err as Error).message);
    }
  };

  const handleLoadJson = (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileReader = new FileReader();
    const file = e.target.files?.[0];
    if (!file) return;

    fileReader.onload = (event) => {
      try {
        const result = event.target?.result;
        if (typeof result !== 'string') return;
        const parsed = JSON.parse(result);
        
        if (!Array.isArray(parsed)) {
          throw new Error('Data should be an array of program review records');
        }
        
        const isValid = parsed.every((item: any) => item && typeof item === 'object' && ('id' in item) && ('program' in item));
        if (parsed.length > 0 && !isValid) {
          throw new Error('Some records are missing required properties like "id" or "program"');
        }

        showConfirm(
          'Import Programs Backup',
          `Successfully parsed ${parsed.length} records. Would you like to load them and replace all current programs? (All current custom states will be archived to Undo history).`,
          () => {
            saveProgramsToStorage(parsed);
          }
        );
      } catch (err) {
        showAlert('Load Backup Failed', 'Failed to parse or load JSON: ' + (err as Error).message);
      } finally {
        e.target.value = '';
      }
    };
    fileReader.readAsText(file);
  };

  // Toggle complete checkbox
  const handleToggleComplete = (id: string) => {
    const updated = programs.map((p) => {
      if (p.id === id) {
        return { ...p, isCompleted: !p.isCompleted };
      }
      return p;
    });
    saveProgramsToStorage(updated);
  };

  // Toggle PI Submitted status checkbox
  const handleTogglePISubmitted = (id: string) => {
    const updated = programs.map((p) => {
      if (p.id === id) {
        const isCurrentlySubm = !!p.submittedDate;
        return { ...p, submittedDate: isCurrentlySubm ? '' : referenceDateString }; // toggles submission status using dynamic today's date
      }
      return p;
    });
    saveProgramsToStorage(updated);
  };

  // Toggle Reviewed status checkbox
  const handleToggleReviewed = (id: string) => {
    const updated = programs.map((p) => {
      if (p.id === id) {
        const isCurrentlyReviewed = !!p.reviewedDate;
        return { ...p, reviewedDate: isCurrentlyReviewed ? '' : referenceDateString }; // toggles review status using dynamic today's date
      }
      return p;
    });
    saveProgramsToStorage(updated);
  };

  // Handle saving (creating/editing) of a program review record
  const handleSaveProgram = (savedProgram: ProgramReview) => {
    const exists = programs.some((p) => p.id === savedProgram.id);
    let updated: ProgramReview[];
    if (exists) {
      updated = programs.map((p) => (p.id === savedProgram.id ? savedProgram : p));
    } else {
      updated = [savedProgram, ...programs];
    }
    saveProgramsToStorage(updated);
  };

  // Handle soft-deleting a program review cycle
  const handleDeleteProgram = (id: string) => {
    const updated = programs.map((p) => {
      if (p.id === id) {
        return { ...p, isDeleted: true };
      }
      return p;
    });
    saveProgramsToStorage(updated);
  };

  // Handle restoring a program review cycle
  const handleRestoreProgram = (id: string) => {
    const updated = programs.map((p) => {
      if (p.id === id) {
        return { ...p, isDeleted: false };
      }
      return p;
    });
    saveProgramsToStorage(updated);
  };

  const handleStatFilterChange = (filterType: string | null) => {
    if (!filterType) {
      setActiveStatFilter(null);
      setCompletionFilter('all');
      return;
    }

    if (activeStatFilter === filterType) {
      setActiveStatFilter(null);
      setCompletionFilter('all');
    } else {
      setActiveStatFilter(filterType);
      if (filterType === 'pending_submission' || filterType === 'pending_review' || filterType === 'pending_approval') {
        setCompletionFilter('pending');
      } else if (filterType === 'approved') {
        setCompletionFilter('completed');
      } else {
        setActiveStatFilter(null);
        setCompletionFilter('all');
      }
    }
  };

  // Reset to original Google Sheet values (Deep Copy to isolate references)
  const handleResetToDefault = () => {
    showConfirm(
      'Reset All Records',
      'Are you sure you want to reset all records back to the default spreadsheet values? All custom edits will be cleared.',
      () => {
        saveProgramsToStorage(JSON.parse(JSON.stringify(INITIAL_PROGRAMS)));
      }
    );
  };

  // Clear all programs with undo backup
  const handleClearAll = () => {
    showConfirm(
      'Clear All Records',
      'Are you sure you want to clear all program records? This will empty the dashboard. You can restore them with the "Undo" button (Ctrl+Z) if needed.',
      () => {
        saveProgramsToStorage([]);
      }
    );
  };

  // Open modal for editing
  const handleEditProgram = (p: ProgramReview) => {
    setSelectedProgram(p);
    setIsModalOpen(true);
  };

  // Update dynamic program fields inline (e.g. notes or STScI scraper refreshes)
  const handleUpdateProgramFields = (idOrIds: string | string[], fields: Partial<ProgramReview>) => {
    let updatedFields = { ...fields };
    if ('approvedDate' in fields) {
      updatedFields.isCompleted = !!fields.approvedDate && fields.approvedDate.trim() !== '';
    }

    const idsToUpdate = Array.isArray(idOrIds) ? idOrIds : [idOrIds];

    const updated = programs.map((p) => {
      if (idsToUpdate.includes(p.id)) {
        return { ...p, ...updatedFields };
      }
      return p;
    });

    saveProgramsToStorage(updated, false, programs);
  };

  // Check for duplicates count
  const duplicateCount = (() => {
    const seen = new Set<string>();
    let count = 0;
    for (const p of programs) {
      if (p.isDeleted) continue;
      const key = `${p.program}-${p.observation}`;
      if (seen.has(key)) {
        count++;
      } else {
        seen.add(key);
      }
    }
    return count;
  })();

  const handleRemoveDuplicates = () => {
    const seen = new Set<string>();
    const deduplicated: ProgramReview[] = [];
    for (const p of programs) {
      const key = `${p.program}-${p.observation}`;
      if (!p.isDeleted && seen.has(key)) {
        continue;
      }
      if (!p.isDeleted) {
        seen.add(key);
      }
      deduplicated.push(p);
    }
    const diff = programs.length - deduplicated.length;
    if (diff > 0) {
      saveProgramsToStorage(deduplicated);
      showAlert('Duplicates Removed', `Successfully removed ${diff} duplicate observation rows!`);
    } else {
      showAlert('No Duplicates', "No active duplicate entries found.");
    }
  };

  // ─── FILTER & SORT PROCESSING ───
  let processedPrograms = [...programs];

  // 1. Text Search filtering (matches PI, APT prep, program number, or observation selection)
  if (searchTerm.trim()) {
    const term = searchTerm.toLowerCase();
    processedPrograms = processedPrograms.filter(
      (p) =>
        (p.pi || '').toLowerCase().includes(term) ||
        (p.aptPrep || '').toLowerCase().includes(term) ||
        (p.program || '').toLowerCase().includes(term) ||
        (p.observation || '').toLowerCase().includes(term)
    );
  }

  // 2. Completion State & Soft-Delete Filtering
  if (completionFilter === 'deleted') {
    processedPrograms = processedPrograms.filter((p) => p.isDeleted);
  } else {
    // Hide soft-deleted records in other views
    processedPrograms = processedPrograms.filter((p) => !p.isDeleted);
    
    if (completionFilter === 'completed') {
      processedPrograms = processedPrograms.filter((p) => p.isCompleted);
    } else if (completionFilter === 'pending') {
      processedPrograms = processedPrograms.filter((p) => !p.isCompleted);
    }
  }

  // 2.3 Interactive Stat-Card Filtering
  if (activeStatFilter) {
    if (activeStatFilter === 'pending_submission') {
      processedPrograms = processedPrograms.filter(
        (p) => !p.isCompleted && (!p.submittedDate || p.submittedDate.trim() === '')
      );
    } else if (activeStatFilter === 'pending_review') {
      processedPrograms = processedPrograms.filter(
        (p) => !p.isCompleted && p.submittedDate && p.submittedDate.trim() !== '' && (!p.reviewedDate || p.reviewedDate.trim() === '')
      );
    } else if (activeStatFilter === 'pending_approval') {
      processedPrograms = processedPrograms.filter(
        (p) => !p.isCompleted && p.submittedDate && p.submittedDate.trim() !== '' && p.reviewedDate && p.reviewedDate.trim() !== ''
      );
    } else if (activeStatFilter === 'approved') {
      processedPrograms = processedPrograms.filter((p) => p.isCompleted);
    }
  }

  // 2.5 Assigned Reviewer Filtering
  if (reviewerFilterEnabled && reviewerName.trim()) {
    const filterName = reviewerName.toLowerCase().trim();
    processedPrograms = processedPrograms.filter((p) => {
      return (
        (p.aptPrep || '').toLowerCase().includes(filterName) ||
        (p.nirspecReviewer || '').toLowerCase().includes(filterName) ||
        (p.nircamReviewer || '').toLowerCase().includes(filterName) ||
        (p.miriReviewer || '').toLowerCase().includes(filterName) ||
        (p.nirissReviewer || '').toLowerCase().includes(filterName)
      );
    });
  }

  // 3. Sorting Execution based on user selection
  processedPrograms.sort((a, b) => {
    let valA: any = '';
    let valB: any = '';
    const key = sortConfig.key;

    if (key === 'program') {
      valA = parseInt(a.program, 10) || 0;
      valB = parseInt(b.program, 10) || 0;
    } else if (key === 'observation') {
      // Parse numeric observation codes like "1" or "14" if possible
      const numA = parseInt(a.observation, 10);
      const numB = parseInt(b.observation, 10);
      valA = isNaN(numA) ? (a.observation || '').toLowerCase() : numA;
      valB = isNaN(numB) ? (b.observation || '').toLowerCase() : numB;
    } else if (key === 'pi') {
      const piA = formatPiName(a.pi || '');
      const piB = formatPiName(b.pi || '');
      valA = (piA.last || '').toLowerCase() + ' ' + (piA.first || '').toLowerCase();
      valB = (piB.last || '').toLowerCase() + ' ' + (piB.first || '').toLowerCase();
    } else if (key === 'aptPrep') {
      valA = (a.aptPrep || '').toLowerCase();
      valB = (b.aptPrep || '').toLowerCase();
    } else if (key === 'obsEarliest') {
      const dateA = parseSheetDate(a.obsEarliest);
      const dateB = parseSheetDate(b.obsEarliest);
      valA = dateA ? dateA.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
      valB = dateB ? dateB.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
    } else if (key === 'obsLatest') {
      const dateA = parseSheetDate(a.obsLatest);
      const dateB = parseSheetDate(b.obsLatest);
      valA = dateA ? dateA.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
      valB = dateB ? dateB.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
    } else if (key === 'flightReadyEarliest' || key === 'flightReady') {
      const dateA = parseSheetDate(a.flightReadyEarliest);
      const dateB = parseSheetDate(b.flightReadyEarliest);
      valA = dateA ? dateA.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
      valB = dateB ? dateB.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
    } else if (key === 'submissionDueDate') {
      const dateA = parseSheetDate(a.submissionDueDate);
      const dateB = parseSheetDate(b.submissionDueDate);
      valA = dateA ? dateA.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
      valB = dateB ? dateB.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
    } else if (key === 'finalizeDueDate') {
      const dateA = parseSheetDate(a.finalizeDueDate);
      const dateB = parseSheetDate(b.finalizeDueDate);
      valA = dateA ? dateA.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
      valB = dateB ? dateB.getTime() : (sortConfig.direction === 'asc' ? Infinity : -Infinity);
    } else if (key === 'submittedDate' || key === 'piSubm') {
      valA = a.submittedDate ? 1 : 0;
      valB = b.submittedDate ? 1 : 0;
    } else if (key === 'reviewedDate' || key === 'reviewed') {
      valA = a.reviewedDate ? 1 : 0;
      valB = b.reviewedDate ? 1 : 0;
    } else if (key === 'isCompleted' || key === 'closed') {
      valA = a.isCompleted ? 1 : 0;
      valB = b.isCompleted ? 1 : 0;
    } else {
      valA = String((a as any)[key] || '').toLowerCase();
      valB = String((b as any)[key] || '').toLowerCase();
    }

    if (valA === valB) return 0;
    if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
    if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
    return 0;
  });

  return (
    <div id="app-root" className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans flex flex-col justify-between">
      <div className="flex-1 w-full mx-auto p-3 sm:p-4">
        
        {/* Navigation Banner Header */}
        <header id="app-header" className="py-2.5 bg-white border border-slate-200 rounded-lg flex flex-col md:flex-row md:items-center justify-between px-4 mb-4 gap-3 shrink-0 z-10 shadow-sm">
          {/* Left Branding and Title */}
          <div className="flex items-center gap-3 md:w-1/3">
            <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center text-white font-black text-xs shadow-sm flex-shrink-0">JW</div>
            <div>
              <h1 className="font-extrabold text-sm sm:text-base tracking-tight text-slate-950 leading-tight">
                NIRSpec MOS APT Program Review Dashboard
              </h1>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                Plan Windows and Due Dates for Submissions and Reviews
              </p>
            </div>
          </div>

          {/* Center: Dynamic Reference Date */}
          <div className="flex justify-start md:justify-center items-center md:w-1/3 py-0.5 md:py-0">
            <div className="flex items-center gap-2 text-xs md:text-sm text-slate-600 font-bold tracking-tight">
              <Calendar className="w-4 h-4 text-blue-600" />
              <span>Today: {REFERENCE_DATE.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}</span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 md:justify-end md:w-1/3">

            {/* Undo Action */}
            <button
              onClick={handleUndo}
              disabled={past.length === 0}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 border rounded shadow-2xs text-[11px] font-bold cursor-pointer transition-all ${
                past.length > 0 
                  ? 'bg-white border-slate-300 text-slate-705 hover:bg-slate-50 hover:text-slate-950 hover:border-slate-400' 
                  : 'bg-slate-50 border-slate-250 text-slate-300 cursor-not-allowed shadow-none'
              }`}
              title="Undo last edit (Ctrl+Z)"
            >
              <Undo className="w-3.5 h-3.5" />
              <span>Undo</span>
            </button>

            {/* Redo Action */}
            <button
              onClick={handleRedo}
              disabled={future.length === 0}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 border rounded shadow-2xs text-[11px] font-bold cursor-pointer transition-all ${
                future.length > 0 
                  ? 'bg-white border-slate-300 text-slate-705 hover:bg-slate-50 hover:text-slate-950 hover:border-slate-400' 
                  : 'bg-slate-50 border-slate-250 text-slate-300 cursor-not-allowed shadow-none'
              }`}
              title="Redo previous edit (Ctrl+Y)"
            >
              <Redo className="w-3.5 h-3.5" />
              <span>Redo</span>
            </button>

            {/* Load Data Action */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 border border-slate-300 bg-white text-slate-705 hover:bg-slate-50 hover:text-slate-950 hover:border-slate-400 rounded shadow-2xs text-[11px] font-bold cursor-pointer transition-all animate-none"
              title="Load program review records from a backup file"
            >
              <Upload className="w-3.5 h-3.5 text-slate-500" />
              <span>Load</span>
            </button>
            
            {/* Save Data Action */}
            <button
              onClick={handleSaveJson}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 border border-slate-300 bg-white text-slate-705 hover:bg-slate-50 hover:text-slate-950 hover:border-slate-400 rounded shadow-2xs text-[11px] font-bold cursor-pointer transition-all animate-none"
              title="Save all program review records to a file"
            >
              <Download className="w-3.5 h-3.5 text-slate-500" />
              <span>Save</span>
            </button>
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleLoadJson} 
              accept=".json" 
              className="hidden" 
            />

            {/* Clear All Programs Action */}
            <button
              onClick={handleClearAll}
              disabled={programs.length === 0}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 border rounded shadow-2xs text-[11px] font-bold cursor-pointer transition-all ${
                programs.length > 0
                  ? 'bg-white border-rose-200 text-rose-700 hover:bg-rose-50 hover:text-rose-900 hover:border-rose-300'
                  : 'bg-slate-50 border-slate-250 text-slate-300 cursor-not-allowed shadow-none'
              }`}
              title="Clear all program records from the dashboard (can be undone)"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear all</span>
            </button>
          </div>
        </header>

        {/* Informational Alert Strip */}
        <div className="flex gap-2.5 bg-amber-50 border-l-4 border-amber-500 p-3 rounded-r mb-4 text-[11px] text-amber-850 shadow-sm">
          <Info className="w-4 h-4 text-amber-600 flex-shrink-0 relative top-0.5" />
          <div className="space-y-1">
            <p className="font-extrabold text-amber-900 uppercase tracking-wide text-[9px]">Submission and Review Deadlines prior to Plan Window Start Date:</p>
            <ul className="list-disc list-inside font-semibold text-amber-805 leading-normal space-y-1 pl-1">
              <li>
                <strong className="font-extrabold text-amber-950">8 weeks (56 – 62 days on a Monday):</strong> PI Submission Due (including full catalog at assigned roll angle)
              </li>
              <li>
                <strong className="font-extrabold text-amber-950">6 weeks (42 days):</strong> CS Review Due
              </li>
              <li>
                <strong className="font-extrabold text-amber-950">4 weeks (28 days):</strong> Finalize APT submission after all iterations with PI
              </li>
              <li>
                <strong className="font-extrabold text-amber-950">2 weeks (11 – 18 days on a Wednesday):</strong> Flight Ready
              </li>
            </ul>
          </div>
        </div>

        {/* Metrics Status Grid */}
        <ReviewStats
          programs={programs.filter(p => !p.isDeleted)}
          referenceDate={REFERENCE_DATE}
          activeStatFilter={activeStatFilter}
          onStatFilterChange={handleStatFilterChange}
        />

        {/* Global Directory Filters and Controls */}
        <ReviewControls
          completionFilter={completionFilter}
          setCompletionFilter={(f) => {
            setCompletionFilter(f);
            setActiveStatFilter(null);
          }}
          onAddProgram={() => setAddProgramOpen(true)}
          reviewerFilterEnabled={reviewerFilterEnabled}
          setReviewerFilterEnabled={setReviewerFilterEnabled}
          reviewerName={reviewerName}
          setReviewerName={setReviewerName}
          columns={columns}
          setColumns={setColumns}
          duplicateCount={duplicateCount}
          onRemoveDuplicates={handleRemoveDuplicates}
          rowPitch={rowPitch}
          setRowPitch={setRowPitch}
          colPitch={colPitch}
          setColPitch={setColPitch}
        />

        {/* Multi-Visualizer Directory Panels */}
        <main id="app-main-content">
          <ReviewTable
            programs={processedPrograms}
            referenceDate={REFERENCE_DATE}
            sortConfig={sortConfig}
            onSortConfigChange={setSortConfig}
            onToggleComplete={handleToggleComplete}
            onTogglePISubmitted={handleTogglePISubmitted}
            onToggleReviewed={handleToggleReviewed}
            onEdit={handleEditProgram}
            onDelete={handleDeleteProgram}
            onRestore={handleRestoreProgram}
            onUpdateProgramFields={handleUpdateProgramFields}
            columns={columns}
            setColumns={setColumns}
            reviewerName={reviewerName}
            rowPitch={rowPitch}
            colPitch={colPitch}
          />
        </main>

        {/* Edit or Create Record Modal */}
        <EditReviewModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          program={selectedProgram}
          onSave={handleSaveProgram}
        />

        {/* Add Program STScI importer Modal */}
        <AddProgramModal
          isOpen={addProgramOpen}
          onClose={() => setAddProgramOpen(false)}
          programs={programs}
          onAddPrograms={(newProgs) => {
            // Add them, skipping exact program + observation duplicates to avoid pollution
            const filteredNew = newProgs.filter(
              (np) => !programs.some((ep) => ep.program === np.program && ep.observation === np.observation)
            );
            // If all are copies but they still want to import, import anyway, otherwise insert
            const listToInsert = filteredNew.length > 0 ? filteredNew : newProgs;
            saveProgramsToStorage([...listToInsert, ...programs]);
          }}
        />

        {/* Custom Confirmation / Alert Dialog for Sandbox/Iframe support */}
        {dialog && dialog.isOpen && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fade-in">
            <div className="bg-white rounded-lg shadow-xl border border-slate-250 max-w-md w-full overflow-hidden transform scale-100 transition-all">
              {/* Header */}
              <div className="bg-slate-50 px-4 py-3 border-b border-slate-200 flex items-center justify-between">
                <h3 className="font-extrabold text-xs sm:text-sm text-slate-900 flex items-center gap-2 uppercase tracking-wide">
                  <Info className="w-4 h-4 text-blue-600" />
                  {dialog.title}
                </h3>
                <button
                  onClick={() => setDialog(null)}
                  className="text-slate-400 hover:text-slate-600 focus:outline-none font-bold text-lg cursor-pointer"
                >
                  ×
                </button>
              </div>
              {/* Body */}
              <div className="p-5 text-xs font-semibold text-slate-600 leading-relaxed">
                {dialog.message}
              </div>
              {/* Footer */}
              <div className="bg-slate-50 px-4 py-3 border-t border-slate-200 flex justify-end gap-2">
                {dialog.type === 'confirm' ? (
                  <>
                    <button
                      onClick={() => setDialog(null)}
                      className="px-3.5 py-2 border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 font-extrabold rounded text-xs cursor-pointer transition-all shadow-2xs"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => {
                        if (dialog.onConfirm) dialog.onConfirm();
                        setDialog(null);
                      }}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white font-extrabold rounded text-xs cursor-pointer transition-all shadow-2xs"
                    >
                      Confirm
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => setDialog(null)}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white font-extrabold rounded text-xs cursor-pointer transition-all shadow-2xs"
                  >
                    OK
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

      </div>

      {/* High Density Footer System Bar */}
      <footer className="h-8 bg-slate-900 text-slate-400 flex items-center px-4 text-[10px] justify-between shrink-0 font-medium">
        <div className="flex gap-4 items-center">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
            Ops Terminal Online
          </div>
          <div className="hidden sm:block">Today&apos;s Date: {REFERENCE_DATE.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</div>
        </div>
        <div className="uppercase tracking-widest font-extrabold text-[9px]">© 2026 Institutional Data Management • STScI & JWST Operations</div>
      </footer>
    </div>
  );
}
