/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from "react";
import {
  ProgramReview,
  SortConfig,
  ColumnConfig,
  DEFAULT_COLUMNS,
} from "../types";
import {
  parseSheetDate,
  getDaysDifference,
  recalculateDates,
  cleanWeekdayAndFormat,
  formatFlightReadyWithYear,
  formatSheetDateWithDay,
} from "../utils/dateHelpers";
import { formatPiName } from "../utils/nameHelpers";
import {
  ExternalLink,
  Edit2,
  Trash2,
  CheckCircle2,
  RefreshCw,
  EyeOff,
} from "lucide-react";

interface ReviewTableProps {
  programs: ProgramReview[];
  referenceDate: Date;
  sortConfig?: SortConfig;
  onSortConfigChange?: (config: SortConfig) => void;
  onToggleComplete: (id: string) => void;
  onTogglePISubmitted: (id: string) => void;
  onToggleReviewed: (id: string) => void;
  onEdit: (program: ProgramReview) => void;
  onDelete: (id: string) => void;
  onRestore: (id: string) => void;
  onUpdateProgramFields?: (
    id: string | string[],
    fields: Partial<ProgramReview>,
  ) => void;
  columns: ColumnConfig[];
  setColumns: React.Dispatch<React.SetStateAction<ColumnConfig[]>>;
  reviewerName?: string;
  rowPitch: number;
  colPitch: number;
}

interface DeferredTextInputProps {
  initialValue: string;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  onCommit: (val: string) => void;
}

function DeferredTextInput({
  initialValue,
  placeholder,
  className,
  disabled,
  onCommit,
}: DeferredTextInputProps) {
  const [value, setValue] = useState(initialValue);

  useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  const handleBlur = () => {
    if (value !== initialValue) {
      onCommit(value);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      setValue(initialValue);
      e.currentTarget.blur();
    }
  };

  return (
    <input
      type="text"
      placeholder={placeholder}
      className={className}
      disabled={disabled}
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
    />
  );
}

const formatWithDayIfPossible = (dateStr: string | undefined): string => {
  if (!dateStr) return "—";
  const trimmed = dateStr.trim();
  if (!trimmed) return "—";
  // If it already has a weekday prefix (e.g., starts with a letter like "Mon"), return it
  if (/^[A-Za-z]{3}/.test(trimmed)) {
    return trimmed;
  }
  const parsed = parseSheetDate(trimmed);
  return parsed ? formatSheetDateWithDay(parsed) : trimmed;
};

export default function ReviewTable({
  programs,
  referenceDate,
  sortConfig,
  onSortConfigChange,
  onToggleComplete,
  onTogglePISubmitted,
  onToggleReviewed,
  onEdit,
  onDelete,
  onRestore,
  onUpdateProgramFields,
  columns,
  setColumns,
  reviewerName = "",
  rowPitch,
  colPitch,
}: ReviewTableProps) {
  // Track rows currently refreshing from STScI
  const [refreshingRows, setRefreshingRows] = useState<Record<string, boolean>>(
    {},
  );
  const [draggedKey, setDraggedKey] = useState<string | null>(null);

  // Column search / filter state
  const [colFilters, setColFilters] = useState<Record<string, string>>({});

  // Derived state: programs filtered by individual column filters
  const filteredPrograms = programs.filter((p) => {
    return Object.keys(colFilters).every((key) => {
      const filterVal = colFilters[key];
      if (!filterVal || filterVal.trim() === "") return true;
      const val = filterVal.toLowerCase().trim();

      switch (key) {
        case "cycle":
          return (p.cycle || "4").toLowerCase().includes(val);
        case "program":
          return (p.program || "").toLowerCase().includes(val);
        case "observation":
          return (
            (p.observation || "").toLowerCase().includes(val) ||
            (p.observation ? false : "all".includes(val))
          );
        case "pi":
          return (p.pi || "").toLowerCase().includes(val);
        case "aptPrep":
          return (p.aptPrep || "").toLowerCase().includes(val);
        case "nirspecReviewer":
          return (p.nirspecReviewer || "").toLowerCase().includes(val);
        case "nircamReviewer":
          return (p.nircamReviewer || "").toLowerCase().includes(val);
        case "miriReviewer":
          return (p.miriReviewer || "").toLowerCase().includes(val);
        case "nirissReviewer":
          return (p.nirissReviewer || "").toLowerCase().includes(val);
        case "notes":
          return (p.notes || "").toLowerCase().includes(val);
        case "submissionDue":
          return (p.submissionDueDate || "").toLowerCase().includes(val);
        case "submittedDate":
          return (p.submittedDate || "").toLowerCase().includes(val);
        case "reviewedDate":
          return (p.reviewedDate || "").toLowerCase().includes(val);
        case "finalizeDue":
          return (p.finalizeDueDate || "").toLowerCase().includes(val);
        case "closed":
          return (p.approvedDate || "").toLowerCase().includes(val);
        case "flightReady": {
          const frStr = p.flightReadyMid
            ? formatFlightReadyWithYear(p.flightReadyMid, p.obsEarliest)
            : "";
          return frStr.toLowerCase().includes(val);
        }
        case "planWindow": {
          const earliest = p.obsEarliest
            ? cleanWeekdayAndFormat(p.obsEarliest)
            : "";
          const latest = p.obsLatest ? cleanWeekdayAndFormat(p.obsLatest) : "";
          const range =
            earliest && latest
              ? `${earliest} - ${latest}`
              : earliest || latest || "";
          return range.toLowerCase().includes(val);
        }
        default:
          return true;
      }
    });
  });

  // Selection states
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [lastSelectedId, setLastSelectedId] = useState<string | null>(null);

  // Row selection logic supporting Cmd/Shift click
  const handleRowSelect = (
    id: string,
    e: React.MouseEvent | React.ChangeEvent,
  ) => {
    const nativeEvent = (e as any).nativeEvent || e;
    const shiftKey = nativeEvent.shiftKey;
    const metaKey = nativeEvent.metaKey || nativeEvent.ctrlKey;

    if (e.type === "click" && shiftKey) {
      if ("preventDefault" in e) {
        e.preventDefault();
      }
    }

    const visibleIds = filteredPrograms.map((p) => p.id);

    if (shiftKey && lastSelectedId && visibleIds.includes(lastSelectedId)) {
      const fromIndex = filteredPrograms.findIndex(
        (p) => p.id === lastSelectedId,
      );
      const toIndex = filteredPrograms.findIndex((p) => p.id === id);
      if (fromIndex !== -1 && toIndex !== -1) {
        const start = Math.min(fromIndex, toIndex);
        const end = Math.max(fromIndex, toIndex);
        const slicedIds = filteredPrograms
          .slice(start, end + 1)
          .map((p) => p.id);

        setSelectedIds((prev) => {
          const nextSet = new Set(prev);
          slicedIds.forEach((sid) => nextSet.add(sid));
          return Array.from(nextSet);
        });
      }
    } else if (metaKey) {
      setSelectedIds((prev) => {
        if (prev.includes(id)) {
          return prev.filter((x) => x !== id);
        } else {
          return [...prev, id];
        }
      });
    } else {
      setSelectedIds((prev) => {
        if (prev.includes(id)) {
          return prev.filter((x) => x !== id);
        } else {
          return [...prev, id];
        }
      });
    }
    setLastSelectedId(id);
  };

  const isAllSelected =
    filteredPrograms.length > 0 &&
    filteredPrograms.every((p) => selectedIds.includes(p.id));

  const handleSelectAllToggle = () => {
    if (isAllSelected) {
      // Clear visible on screen
      const visibleSet = new Set(filteredPrograms.map((p) => p.id));
      setSelectedIds((prev) => prev.filter((id) => !visibleSet.has(id)));
    } else {
      setSelectedIds((prev) => {
        const nextSet = new Set(prev);
        filteredPrograms.forEach((p) => nextSet.add(p.id));
        return Array.from(nextSet);
      });
    }
  };

  const handleFieldUpdate = (id: string, fields: Partial<ProgramReview>) => {
    if (!onUpdateProgramFields) return;
    if (selectedIds.includes(id)) {
      onUpdateProgramFields(selectedIds, fields);
    } else {
      onUpdateProgramFields(id, fields);
    }
  };

  // Strip weekend weekdays from display text
  const stripWeekday = (val: string) => {
    if (!val) return "";
    // e.g. "Sun  7/19" -> "7/19" or "Mon 11/18" -> "11/18"
    return val.replace(/^[A-Za-z]{3}\s+/, "").trim();
  };

  // Helper to map column key to logical data sort fields
  const getSortKey = (key: string): string => {
    if (key === "submissionDue") return "submissionDueDate";
    if (key === "finalizeDue") return "finalizeDueDate";
    if (key === "closed") return "isCompleted";
    if (key === "flightReady") return "flightReadyMid";
    if (key === "planWindow") return "obsEarliest";
    return key;
  };

  const handleSort = (key: string) => {
    if (!onSortConfigChange) return;
    const targetSortKey = getSortKey(key);
    const isSameKey = sortConfig?.key === targetSortKey;
    const direction =
      isSameKey && sortConfig?.direction === "asc" ? "desc" : "asc";
    onSortConfigChange({ key: targetSortKey, direction });
  };

  const renderSortIndicator = (key: string) => {
    if (!sortConfig) return null;
    const targetSortKey = getSortKey(key);
    const isActive = sortConfig.key === targetSortKey;
    if (isActive) {
      return (
        <span className="inline-flex ml-1 text-xs font-black text-blue-600">
          {sortConfig.direction === "asc" ? "▲" : "▼"}
        </span>
      );
    }
    return null;
  };

  // Drag and Drop handlers for column headers to reorder them
  const handleDragStart = (e: React.DragEvent, key: string) => {
    if (key === "actions" || key === "select") return; // actions and select should stay
    setDraggedKey(key);
    e.dataTransfer.setData("text/plain", key);
  };

  const handleDragOver = (e: React.DragEvent, key: string) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, targetKey: string) => {
    e.preventDefault();
    if (!draggedKey || draggedKey === targetKey) return;
    if (targetKey === "actions" || targetKey === "select") return; // stay at endpoints

    const fromIndex = columns.findIndex((c) => c.key === draggedKey);
    const toIndex = columns.findIndex((c) => c.key === targetKey);
    if (fromIndex === -1 || toIndex === -1) return;

    const updated = [...columns];
    const [removed] = updated.splice(fromIndex, 1);
    updated.splice(toIndex, 0, removed);

    setColumns(updated);
    setDraggedKey(null);
  };

  // Resize handler
  const activeResizeRef = useRef<{
    colKey: string;
    startX: number;
    startWidth: number;
  } | null>(null);

  const startResize = (colKey: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const currentCol = columns.find((c) => c.key === colKey);
    if (!currentCol) return;

    activeResizeRef.current = {
      colKey,
      startX: e.clientX,
      startWidth: currentCol.width,
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!activeResizeRef.current) return;
    const { colKey, startX, startWidth } = activeResizeRef.current;
    const deltaX = e.clientX - startX;
    const newWidth = Math.max(35, startWidth + deltaX);

    setColumns((prev) =>
      prev.map((col) =>
        col.key === colKey ? { ...col, width: newWidth } : col,
      ),
    );
  };

  const handleMouseUp = () => {
    activeResizeRef.current = null;
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  const resetColumns = () => {
    setColumns(DEFAULT_COLUMNS);
  };

  // Scrape/Refresh an individual program row
  const handleRefreshRow = async (
    programId: string,
    programNum: string,
    obsNum: string,
  ) => {
    if (!onUpdateProgramFields || !programNum) return;

    setRefreshingRows((prev) => ({ ...prev, [programId]: true }));
    try {
      const res = await fetch(`/api/parse-stsci?program=${programNum}`);
      if (!res.ok) throw new Error();
      const data = await res.json();

      if (data.success) {
        const updatePayload: Partial<ProgramReview> = {};

        // Update basic info
        if (data.pi) updatePayload.pi = data.pi;
        if (data.cycle) updatePayload.cycle = data.cycle;
        if (data.aptPrep) updatePayload.aptPrep = data.aptPrep;
        if (data.programInfoUrl)
          updatePayload.programInfoUrl = data.programInfoUrl;
        if (data.visitStatusUrl)
          updatePayload.visitStatusUrl = data.visitStatusUrl;

        // Specific instrument reviewers if returned
        if (data.nirspecReviewer) updatePayload.nirspecReviewer = data.nirspecReviewer;
        if (data.nircamReviewer) updatePayload.nircamReviewer = data.nircamReviewer;
        if (data.miriReviewer) updatePayload.miriReviewer = data.miriReviewer;
        if (data.nirissReviewer) updatePayload.nirissReviewer = data.nirissReviewer;

        const currentObservationString = obsNum ? obsNum.trim() : "";
        const targetObsArray = currentObservationString
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);

        // Find matching observations
        let matchedResults = (data.results || []).filter((r: any) => {
          const rObs = String(r.observation).trim();
          return (
            targetObsArray.includes(rObs) ||
            rObs === currentObservationString
          );
        });

        // Fallback to ALL results if none of the specific observation numbers matched
        if (matchedResults.length === 0 && (data.results || []).length > 0) {
          matchedResults = data.results;
          const sortedObs = [...matchedResults].sort((a: any, b: any) => {
            const na = parseInt(a.observation, 10);
            const nb = parseInt(b.observation, 10);
            if (!isNaN(na) && !isNaN(nb)) return na - nb;
            return String(a.observation).localeCompare(String(b.observation));
          });
          const updatedObsNum = sortedObs.map((r: any) => r.observation).join(",");
          updatePayload.observation = updatedObsNum;
        }

        if (matchedResults.length > 0) {
          let minEarliest: Date | null = null;
          let maxLatest: Date | null = null;
          let minEarliestStr = "";
          let maxLatestStr = "";

          matchedResults.forEach((r: any) => {
            const earD = parseSheetDate(r.obsEarliest);
            const latD = parseSheetDate(r.obsLatest);

            if (earD) {
              if (!minEarliest || earD < minEarliest) {
                minEarliest = earD;
                minEarliestStr = r.obsEarliest;
              }
            }
            if (latD) {
              if (!maxLatest || latD > maxLatest) {
                maxLatest = latD;
                maxLatestStr = r.obsLatest;
              }
            }
          });

          if (minEarliestStr) {
            updatePayload.obsEarliest = minEarliestStr;
            if (maxLatest && minEarliest) {
              const diffMs = (maxLatest as Date).getTime() - (minEarliest as Date).getTime();
              const midTime = (minEarliest as Date).getTime() + diffMs / 2;
              const midDate = new Date(midTime);
              const m = midDate.getMonth() + 1;
              const d = midDate.getDate();
              const yStr = String(midDate.getFullYear() % 100);
              updatePayload.obsMid = `${m}/${d}/${yStr}`;
            }
          }
          if (maxLatestStr) {
            updatePayload.obsLatest = maxLatestStr;
          }

          if (minEarliestStr) {
            const calcs = recalculateDates(minEarliestStr);
            if (calcs.submissionDueDate) {
              updatePayload.submissionDueDate = calcs.submissionDueDate;
              updatePayload.reviewDueDate = calcs.reviewDueDate;
              updatePayload.finalizeDueDate = calcs.finalizeDueDate;
              updatePayload.flightReadyEarliest = calcs.flightReadyEarliest;
              updatePayload.flightReadyMid = calcs.flightReadyMid;
              updatePayload.flightReadyLatest = calcs.flightReadyLatest;
            }
          }
        }

        onUpdateProgramFields(programId, updatePayload);
      }
    } catch (err) {
      console.error(`Failed to refresh program ${programNum}`, err);
    } finally {
      setRefreshingRows((prev) => ({ ...prev, [programId]: false }));
    }
  };

  // Colors based on Status (Submission Status logic merged directly)
  const getSubmitCellStyles = (p: ProgramReview) => {
    const isApproved =
      (p.approvedDate && p.approvedDate.trim() !== "") || p.isCompleted;
    if (isApproved) {
      return "text-slate-900 font-normal";
    }
    if (p.submittedDate && p.submittedDate.trim() !== "") {
      return "text-slate-900 font-normal";
    }
    if (!p.submissionDueDate) return "text-slate-400";
    const subDate = parseSheetDate(p.submissionDueDate);
    if (!subDate) return "text-slate-500";

    const daysDiff = getDaysDifference(referenceDate, subDate);
    if (daysDiff < 0) {
      return "bg-rose-50 text-rose-700 font-bold border-l-2 border-rose-500 animate-pulse";
    } else if (daysDiff <= 30) {
      return "bg-amber-50 text-amber-700 font-semibold border-l-2 border-amber-500";
    }
    return "bg-slate-50 text-slate-600";
  };

  // Colors based on Status (Review Status logic)
  const getReviewCellStyles = (p: ProgramReview) => {
    const isApproved =
      (p.approvedDate && p.approvedDate.trim() !== "") || p.isCompleted;
    if (isApproved) {
      return "text-slate-900 font-normal";
    }
    if (p.reviewedDate && p.reviewedDate.trim() !== "") {
      return "text-slate-900 font-normal";
    }
    if (!p.reviewDueDate) return "text-slate-400";
    const revDate = parseSheetDate(p.reviewDueDate);
    if (!revDate) return "text-slate-500";

    const daysDiff = getDaysDifference(referenceDate, revDate);
    if (daysDiff < 0) {
      return "bg-rose-50 text-rose-700 font-bold border-l-2 border-rose-500";
    } else if (daysDiff <= 30) {
      return "bg-amber-50 text-amber-700 font-semibold border-l-2 border-amber-500";
    }
    return "bg-slate-50 text-slate-600";
  };

  // Colors based on Status (Finalize Status logic merged directly)
  const getFinalizeCellStyles = (p: ProgramReview) => {
    const isApproved =
      (p.approvedDate && p.approvedDate.trim() !== "") || p.isCompleted;
    if (isApproved) {
      return "text-slate-900 font-normal";
    }
    if (!p.finalizeDueDate) return "text-slate-400";
    const finDate = parseSheetDate(p.finalizeDueDate);
    if (!finDate) return "text-slate-500";

    const daysDiff = getDaysDifference(referenceDate, finDate);
    if (daysDiff < 0) {
      return "bg-rose-50 text-rose-700 font-bold border-l-2 border-rose-500";
    } else if (daysDiff <= 30) {
      return "bg-amber-50 text-amber-700 font-semibold border-l-2 border-amber-500";
    }
    return "bg-slate-50 text-slate-600";
  };

  // Colors based on Flight Ready Dates (unless APPROVED)
  const getFlightReadyCellStyles = (p: ProgramReview) => {
    const isApproved =
      (p.approvedDate && p.approvedDate.trim() !== "") || p.isCompleted;
    if (isApproved) {
      return "text-slate-900 font-normal";
    }
    if (!p.flightReadyMid) return "text-slate-400";

    // Crucial bug fix: use the helper to get the date string with correct year before parsing!
    const flightReadyWithYear = formatFlightReadyWithYear(
      p.flightReadyMid,
      p.obsEarliest,
    );
    const frDate = parseSheetDate(flightReadyWithYear);
    if (!frDate) return "text-slate-500";

    const daysDiff = getDaysDifference(referenceDate, frDate);
    if (daysDiff < 0) {
      return "bg-rose-50 text-rose-700 font-bold border-l-2 border-rose-500";
    } else if (daysDiff <= 30) {
      return "bg-amber-50 text-amber-700 font-semibold border-l-2 border-amber-500";
    }
    return "bg-slate-50 text-slate-605 font-medium";
  };

  // Spacing helper maps for Row and Col Pitch levels
  const getRowPaddingClass = (originalType: 'py-2.5' | 'py-2' | 'py-1' = 'py-2.5') => {
    if (originalType === 'py-1') {
      switch (rowPitch) {
        case 1:
          return "py-0.5";
        case 2:
          return "py-0.5";
        case 3:
          return "py-1";
        case 4:
          return "py-1";
        case 5:
          return "py-1.5";
        default:
          return "py-0.5";
      }
    } else if (originalType === 'py-2') {
      switch (rowPitch) {
        case 1:
          return "py-0.5";
        case 2:
          return "py-1";
        case 3:
          return "py-1.5";
        case 4:
          return "py-2";
        case 5:
          return "py-2";
        default:
          return "py-0.5";
      }
    } else {
      // py-2.5
      switch (rowPitch) {
        case 1:
          return "py-0.5";
        case 2:
          return "py-1";
        case 3:
          return "py-1.5";
        case 4:
          return "py-2";
        case 5:
          return "py-2.5";
        default:
          return "py-0.5";
      }
    }
  };

  const getColPaddingClass = () => {
    switch (colPitch) {
      case 1:
        return "px-0.5";
      case 2:
        return "px-1";
      case 3:
        return "px-1.5";
      case 4:
        return "px-2";
      case 5:
        return "px-3";
      default:
        return "px-0.5";
    }
  };

  const r25 = getRowPaddingClass('py-2.5');
  const r20 = getRowPaddingClass('py-2');
  const r10 = getRowPaddingClass('py-1');
  const c2 = getColPaddingClass();

  // Build current active column styles/width
  const visibleColumns = columns.filter((c) => c.visible);

  const getColWidth = (col: ColumnConfig) => {
    if (col.key === "select") {
      return col.width + colPitch;
    }
    // Widen columns proportionally to colPitch to ensure inputs and text expand beautifully
    return col.width + colPitch * 3;
  };

  const totalTableWidth = visibleColumns.reduce(
    (acc, curr) => acc + getColWidth(curr),
    0,
  );

  return (
    <div className="space-y-3.5">
      <style>{`
        /* Dynamic table cell spacing based on rowPitch and colPitch */
        .pitch-table th, .pitch-table td {
          padding-top: ${rowPitch}px !important;
          padding-bottom: ${rowPitch}px !important;
          padding-left: ${colPitch}px !important;
          padding-right: ${colPitch}px !important;
        }
        /* Dynamic input spacing inside the table cells (not including checkboxes) */
        .pitch-table input[type="text"], .pitch-table input:not([type="checkbox"]) {
          padding-top: ${Math.max(0, Math.floor(rowPitch * 0.4))}px !important;
          padding-bottom: ${Math.max(0, Math.floor(rowPitch * 0.4))}px !important;
          padding-left: ${Math.max(0, Math.floor(colPitch * 0.5))}px !important;
          padding-right: ${Math.max(0, Math.floor(colPitch * 0.5))}px !important;
          margin: 0 !important;
        }
        /* Gaps inside cell layout containers scale down */
        .pitch-table .gap-1\\.5 {
          gap: ${colPitch > 2 ? '0.375rem' : '0px'} !important;
        }
        .pitch-table .gap-1 {
          gap: ${colPitch > 2 ? '0.25rem' : '0px'} !important;
        }
        /* Action buttons within the table scale padding */
        .pitch-table button, .pitch-table a.inline-flex {
          padding-top: ${Math.max(0, Math.floor(rowPitch * 0.25))}px !important;
          padding-bottom: ${Math.max(0, Math.floor(rowPitch * 0.25))}px !important;
          padding-left: ${Math.max(0, Math.floor(colPitch * 0.4))}px !important;
          padding-right: ${Math.max(0, Math.floor(colPitch * 0.4))}px !important;
        }
        /* Dynamic borders and border-radius on buttons/badges */
        .pitch-table .rounded {
          border-radius: ${rowPitch > 2 ? '4px' : '0px'} !important;
        }
        /* Badge elements inside table */
        .pitch-table span.bg-slate-100 {
          padding-top: ${Math.max(0, Math.floor(rowPitch * 0.2))}px !important;
          padding-bottom: ${Math.max(0, Math.floor(rowPitch * 0.2))}px !important;
          padding-left: ${Math.max(0, Math.floor(colPitch * 0.3))}px !important;
          padding-right: ${Math.max(0, Math.floor(colPitch * 0.3))}px !important;
        }
      `}</style>

      {/* Main Table Container */}
      <div className="bg-white rounded border border-slate-200 overflow-hidden shadow-xs">
        <div className="overflow-x-auto">
          <table
            className="text-left border-collapse pitch-table"
            style={{ width: totalTableWidth, tableLayout: "fixed" }}
          >
            <colgroup>
              {visibleColumns.map((col) => (
                <col key={col.key} style={{ width: getColWidth(col) }} />
              ))}
            </colgroup>
            <thead>
              <tr className="bg-slate-50 text-[10px] uppercase tracking-widest font-bold text-slate-500 border-b border-slate-200 select-none">
                {visibleColumns.map((col) => {
                  const isActions = col.key === "actions";
                  const isSelect = col.key === "select";
                  const isDraggable = !isActions && !isSelect;

                  if (isSelect) {
                    return (
                      <th
                        key="select"
                        className={`${r25} ${c2} border-r border-slate-200 text-center select-none`}
                        style={{ width: getColWidth(col) }}
                      >
                        <input
                          type="checkbox"
                          checked={isAllSelected}
                          onChange={handleSelectAllToggle}
                          className="w-3.5 h-3.5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                          title="Select all / none"
                        />
                      </th>
                    );
                  }

                  const customHeaderTitle =
                    col.key === "submissionDue"
                      ? "Due 8 weeks (56 – 62 days on a Monday) before Plan Window start. (Includes full catalog at assigned roll angle)."
                      : col.key === "reviewDue"
                        ? "Due 6 weeks (42 days) before Plan Window start."
                        : col.key === "finalizeDue"
                          ? "Due 4 weeks (28 days) before Plan Window start. (Finalize APT submission after all iterations with PI)."
                          : col.key === "flightReady"
                            ? "Due 2 weeks (11 – 18 days on a Wednesday) before Plan Window start."
                            : col.key === "planWindow"
                              ? "Schedules of earliest & latest observations."
                              : "";
                  const thTitle = customHeaderTitle
                    ? `${customHeaderTitle}${isDraggable ? " \n\n• Drag to reorder column\n• Click to sort\n• Double-click resize bar to reset" : ""}`
                    : isDraggable
                      ? "Drag to reorder column • Click to sort • Double-click resize bar to reset"
                      : "";

                  return (
                    <th
                      key={col.key}
                      draggable={isDraggable}
                      onDragStart={(e) => handleDragStart(e, col.key)}
                      onDragOver={(e) => handleDragOver(e, col.key)}
                      onDrop={(e) => handleDrop(e, col.key)}
                      className={`relative ${r25} ${c2} border-r border-slate-200 group overflow-hidden ${
                        draggedKey === col.key
                          ? "bg-blue-50/50 border-r-dashed"
                          : ""
                      } ${isDraggable ? "cursor-grab active:cursor-grabbing" : ""}`}
                    >
                      <div
                        onClick={() => isDraggable && handleSort(col.key)}
                        className={`flex items-center justify-between w-full h-full text-[10px] tracking-wider transition-colors pr-2 ${
                          isDraggable
                            ? "cursor-pointer hover:text-blue-700"
                            : ""
                        }`}
                        title={thTitle}
                      >
                        <span className="truncate">{col.label}</span>
                        {isDraggable && renderSortIndicator(col.key)}
                      </div>

                      {/* Column Resize Handle */}
                      <div
                        className="absolute right-0 top-0 bottom-0 w-1 bg-slate-300 opacity-0 group-hover:opacity-100 cursor-col-resize z-20 transition-all hover:bg-blue-500 hover:w-1.5 active:bg-blue-700 active:w-1.5"
                        onMouseDown={(e) => startResize(col.key, e)}
                        onDoubleClick={() => {
                          const defCol = DEFAULT_COLUMNS.find(
                            (c) => c.key === col.key,
                          );
                          if (defCol) {
                            setColumns((prev) =>
                              prev.map((c) =>
                                c.key === col.key
                                  ? { ...c, width: defCol.width }
                                  : c,
                              ),
                            );
                          }
                        }}
                      />
                    </th>
                  );
                })}
              </tr>
              {/* Column-by-column Filter Input Row */}
              <tr className="bg-slate-100/70 border-b border-slate-200">
                {visibleColumns.map((col) => {
                  const isSelect = col.key === "select";
                  const isActions = col.key === "actions";

                  if (isSelect) {
                    return (
                      <td
                        key="filter-select"
                        className={`${r10} ${c2} border-r border-slate-200 text-center text-[10px] text-slate-400 select-none`}
                      >
                        🔍
                      </td>
                    );
                  }

                  if (isActions) {
                    const hasFilters = Object.values(colFilters).some(
                      (v) => typeof v === "string" && v.trim() !== "",
                    );
                    return (
                      <td
                        key="filter-actions"
                        className={`${r10} ${c2} text-center`}
                      >
                        {hasFilters && (
                          <button
                            type="button"
                            onClick={() => setColFilters({})}
                            className="bg-rose-50 hover:bg-rose-100 hover:text-rose-800 text-rose-700 border border-rose-250 rounded font-black text-[9px] uppercase tracking-wider px-1.5 py-0.5 shadow-2xs leading-none transition-colors cursor-pointer"
                            title="Clear all column filters"
                          >
                            Clear
                          </button>
                        )}
                      </td>
                    );
                  }

                  // Placeholder text based on column key
                  let ph = "Filter...";
                  if (col.key === "program") ph = "id...";
                  else if (col.key === "pi") ph = "name...";
                  else if (col.key === "notes") ph = "note...";

                  return (
                    <td
                      key={`filter-${col.key}`}
                      className={`${r10} ${c2} border-r border-slate-200 align-middle`}
                    >
                      <input
                        type="text"
                        placeholder={ph}
                        className="w-full px-1.5 py-1 bg-white border border-slate-200 hover:border-slate-300 focus:border-blue-400 focus:ring-1 focus:ring-blue-400/20 rounded text-[11px] placeholder:text-slate-300 font-medium text-slate-700 outline-none transition-all shadow-2xs"
                        value={colFilters[col.key] || ""}
                        onChange={(e) => {
                          setColFilters((prev) => ({
                            ...prev,
                            [col.key]: e.target.value,
                          }));
                        }}
                      />
                    </td>
                  );
                })}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs">
              {filteredPrograms.length === 0 ? (
                <tr>
                  <td
                    colSpan={visibleColumns.length}
                    className="py-8 text-center text-slate-405 font-bold text-xs uppercase tracking-wider"
                  >
                    No programs match your current filter criteria.
                  </td>
                </tr>
              ) : (
                filteredPrograms.map((p) => {
                  const formattedPi = formatPiName(p.pi || "");
                  const displayFlightReady = p.flightReadyMid
                    ? stripWeekday(p.flightReadyMid)
                    : "—";
                  const isDeleted = !!p.isDeleted;
                  const isSelected = selectedIds.includes(p.id);

                  const filterName = reviewerName.trim().toLowerCase();
                  const isAssigned =
                    !filterName ||
                    (p.aptPrep || "").toLowerCase().includes(filterName) ||
                    (p.nirspecReviewer || "")
                      .toLowerCase()
                      .includes(filterName) ||
                    (p.nircamReviewer || "")
                      .toLowerCase()
                      .includes(filterName) ||
                    (p.miriReviewer || "").toLowerCase().includes(filterName) ||
                    (p.nirissReviewer || "").toLowerCase().includes(filterName);

                  return (
                    <tr
                      key={p.id}
                      className={`hover:bg-blue-100/45 transition-colors ${
                        isSelected
                          ? "bg-blue-50/70 font-medium text-blue-900 border-l-2 border-blue-600"
                          : isDeleted
                            ? "bg-rose-50/25 text-slate-400 font-medium"
                            : !isAssigned
                              ? "bg-slate-200/85 text-slate-500 opacity-90"
                              : p.isCompleted
                                ? "bg-slate-50/40 text-slate-405"
                                : "text-slate-700"
                      }`}
                    >
                      {visibleColumns.map((col) => {
                        switch (col.key) {
                          case "select":
                            return (
                              <td
                                key="select"
                                className={`${r25} ${c2} text-center border-r border-slate-100 transition-colors cursor-pointer select-none`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleRowSelect(p.id, e);
                                }}
                              >
                                <input
                                  type="checkbox"
                                  checked={isSelected}
                                  onChange={(e) => {
                                    e.stopPropagation();
                                    handleRowSelect(p.id, e);
                                  }}
                                  className="w-3.5 h-3.5 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                                />
                              </td>
                            );

                          case "cycle":
                            return (
                              <td
                                key="cycle"
                                className={`${r25} ${c2} font-black text-slate-505 border-r border-slate-50 text-center truncate`}
                              >
                                {p.cycle || "4"}
                              </td>
                            );

                          case "program":
                            return (
                              <td
                                key="program"
                                className={`${r25} ${c2} font-extrabold tracking-tight border-r border-slate-50 truncate`}
                              >
                                <a
                                  href={
                                    p.programInfoUrl ||
                                    `https://www.stsci.edu/jwst/science-execution/program-information?id=${p.program}`
                                  }
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:text-blue-800 hover:underline cursor-pointer"
                                  title="STScI Program Info Sheet"
                                >
                                  {p.program}
                                </a>
                              </td>
                            );

                          case "observation":
                            return (
                              <td
                                key="observation"
                                className={`${r25} ${c2} font-mono text-[11px] border-r border-slate-50 truncate`}
                              >
                                {p.observation ? (
                                  <span className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-600 font-semibold break-all">
                                    {p.observation}
                                  </span>
                                ) : (
                                  <span className="text-slate-400 italic">
                                    All
                                  </span>
                                )}
                              </td>
                            );

                          case "pi":
                            return (
                              <td
                                key="pi"
                                className={`${r25} ${c2} text-slate-900 border-r border-slate-50 truncate relative group`}
                              >
                                <DeferredTextInput
                                  className="w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-xs font-bold focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-slate-800"
                                  placeholder=""
                                  initialValue={p.pi || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, { pi: val })
                                  }
                                />
                              </td>
                            );

                          case "aptPrep":
                            return (
                              <td
                                key="aptPrep"
                                className={`${r25} ${c2} text-slate-500 font-medium truncate border-r border-slate-50 relative group`}
                              >
                                <DeferredTextInput
                                  className="w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-xs focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-slate-700"
                                  placeholder=""
                                  initialValue={p.aptPrep || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, { aptPrep: val })
                                  }
                                />
                              </td>
                            );

                          case "nirspecReviewer":
                            return (
                              <td
                                key="nirspecReviewer"
                                className={`${r25} ${c2} text-slate-500 font-medium truncate border-r border-slate-50 relative group`}
                              >
                                <DeferredTextInput
                                  className="w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-xs focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-slate-700"
                                  placeholder=""
                                  initialValue={p.nirspecReviewer || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, {
                                      nirspecReviewer: val,
                                    })
                                  }
                                />
                              </td>
                            );

                          case "nircamReviewer":
                            return (
                              <td
                                key="nircamReviewer"
                                className={`${r25} ${c2} text-slate-500 font-medium truncate border-r border-slate-50 relative group`}
                              >
                                <DeferredTextInput
                                  className="w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-xs focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-slate-700"
                                  placeholder=""
                                  initialValue={p.nircamReviewer || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, {
                                      nircamReviewer: val,
                                    })
                                  }
                                />
                              </td>
                            );

                          case "miriReviewer":
                            return (
                              <td
                                key="miriReviewer"
                                className={`${r25} ${c2} text-slate-500 font-medium truncate border-r border-slate-50 relative group`}
                              >
                                <DeferredTextInput
                                  className="w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-xs focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-slate-700"
                                  placeholder=""
                                  initialValue={p.miriReviewer || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, {
                                      miriReviewer: val,
                                    })
                                  }
                                />
                              </td>
                            );

                          case "nirissReviewer":
                            return (
                              <td
                                key="nirissReviewer"
                                className={`${r25} ${c2} text-slate-500 font-medium truncate border-r border-slate-50 relative group`}
                              >
                                <DeferredTextInput
                                  className="w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-xs focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-slate-700"
                                  placeholder=""
                                  initialValue={p.nirissReviewer || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, {
                                      nirissReviewer: val,
                                    })
                                  }
                                />
                              </td>
                            );

                          case "notes":
                            return (
                              <td
                                key="notes"
                                className={`${r10} ${c2} border-r border-slate-50 relative group`}
                              >
                                <DeferredTextInput
                                  className="w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-xs focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-slate-700"
                                  placeholder=""
                                  initialValue={p.notes || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, { notes: val })
                                  }
                                />
                              </td>
                            );

                          case "submissionDue":
                            return (
                              <td
                                key="submissionDue"
                                title="Due 8 weeks (56 – 62 days on a Monday) before Plan Window start (includes full catalog at assigned roll angle)"
                                className={`font-mono text-[11.5px] font-bold border-r border-slate-50 truncate text-center ${r25} ${c2} ${getSubmitCellStyles(p)}`}
                              >
                                {formatWithDayIfPossible(p.submissionDueDate)}
                              </td>
                            );

                          case "submittedDate":
                            const hasSubmitted =
                              p.submittedDate && p.submittedDate.trim() !== "";
                            return (
                              <td
                                key="submittedDate"
                                className={`border-r border-slate-50 relative group transition-all duration-200 ${r10} ${c2} ${
                                  hasSubmitted
                                    ? "bg-emerald-50/80 border-l-2 border-emerald-500"
                                    : ""
                                }`}
                              >
                                <DeferredTextInput
                                  className={`w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-[11px] font-mono font-bold focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-center ${
                                    hasSubmitted
                                      ? "text-emerald-800"
                                      : "text-slate-700"
                                  }`}
                                  placeholder="M/D/YY"
                                  initialValue={p.submittedDate || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, {
                                      submittedDate: val,
                                    })
                                  }
                                />
                              </td>
                            );

                          case "reviewDue":
                            return (
                              <td
                                key="reviewDue"
                                title="Due 6 weeks (42 days) before Plan Window start"
                                className={`font-mono text-[11.5px] font-bold border-r border-slate-50 truncate text-center ${r25} ${c2} ${getReviewCellStyles(p)}`}
                              >
                                {formatWithDayIfPossible(p.reviewDueDate)}
                              </td>
                            );

                          case "finalizeDue":
                            return (
                              <td
                                key="finalizeDue"
                                title="Due 4 weeks (28 days) before Plan Window start"
                                className={`font-mono text-[11.5px] font-bold border-r border-slate-50 truncate text-center ${r25} ${c2} ${getFinalizeCellStyles(p)}`}
                              >
                                 {formatWithDayIfPossible(p.finalizeDueDate)}
                              </td>
                            );

                          case "reviewedDate":
                            const hasReviewed =
                              p.reviewedDate && p.reviewedDate.trim() !== "";
                            return (
                              <td
                                key="reviewedDate"
                                className={`border-r border-slate-50 relative group transition-all duration-200 ${r10} ${c2} ${
                                  hasReviewed
                                    ? "bg-emerald-50/80 border-l-2 border-emerald-500"
                                    : ""
                                }`}
                              >
                                <DeferredTextInput
                                  className={`w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-[11px] font-mono font-bold focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-center ${
                                    hasReviewed
                                      ? "text-emerald-800"
                                      : "text-slate-700"
                                  }`}
                                  placeholder="M/D/YY"
                                  initialValue={p.reviewedDate || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, {
                                      reviewedDate: val,
                                    })
                                  }
                                />
                              </td>
                            );

                          case "flightReady":
                            const cleanFlightReady = p.flightReadyMid
                              ? formatFlightReadyWithYear(
                                  p.flightReadyMid,
                                  p.obsEarliest,
                                )
                              : "—";
                            return (
                              <td
                                key="flightReady"
                                title="Due 2 weeks (11 – 18 days on a Wednesday) before Plan Window start"
                                className={`font-mono text-[11.5px] font-bold border-r border-slate-50 truncate text-center ${r25} ${c2} ${getFlightReadyCellStyles(p)}`}
                              >
                                {cleanFlightReady}
                              </td>
                            );

                          case "planWindow":
                            const earliestStr = p.obsEarliest
                              ? cleanWeekdayAndFormat(p.obsEarliest)
                              : "";
                            const latestStr = p.obsLatest
                              ? cleanWeekdayAndFormat(p.obsLatest)
                              : "";
                            const displayRange =
                              earliestStr && latestStr
                                ? `${earliestStr} - ${latestStr}`
                                : earliestStr || latestStr || "";
                            return (
                              <td
                                key="planWindow"
                                className={`font-mono text-[11px] font-medium text-slate-700 border-r border-slate-50 ${r20} ${c2}`}
                              >
                                <div className="flex items-center justify-between gap-1.5 w-full">
                                  {/* Left Link Icon to visitStatusUrl */}
                                  <div className="flex-shrink-0">
                                    {p.visitStatusUrl ? (
                                      <a
                                        href={p.visitStatusUrl}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center justify-center p-1 bg-teal-55 hover:bg-teal-100 text-teal-700 rounded cursor-pointer transition-colors"
                                        title="STScI Visit Status Information"
                                      >
                                        <ExternalLink className="w-3.5 h-3.5 text-teal-600" />
                                      </a>
                                    ) : (
                                      <span className="w-5" />
                                    )}
                                  </div>

                                  {/* Center Display date range */}
                                  <div className="flex-1 text-center font-bold text-[11.5px] truncate">
                                    {displayRange || (
                                      <span className="text-amber-500 italic font-medium">
                                        Pending
                                      </span>
                                    )}
                                  </div>

                                  {/* Right refresh button */}
                                  <div className="flex-shrink-0">
                                    {!isDeleted && p.program && (
                                      <button
                                        onClick={() =>
                                          handleRefreshRow(
                                            p.id,
                                            p.program,
                                            p.observation || "all",
                                          )
                                        }
                                        disabled={refreshingRows[p.id]}
                                        className="text-slate-400 hover:text-blue-600 transition-colors bg-transparent border-0 cursor-pointer p-0.5"
                                        title="Refresh scheduling windows directly from STScI"
                                      >
                                        <RefreshCw
                                          className={`w-3.5 h-3.5 ${refreshingRows[p.id] ? "animate-spin text-blue-605" : ""}`}
                                        />
                                      </button>
                                    )}
                                  </div>
                                </div>
                              </td>
                            );

                          case "closed":
                            const hasApproved =
                              p.approvedDate && p.approvedDate.trim() !== "";
                            return (
                              <td
                                key="closed"
                                className={`border-r border-slate-50 relative group transition-all duration-200 ${r10} ${c2} ${
                                  hasApproved
                                    ? "bg-emerald-50/80 border-l-2 border-emerald-500"
                                    : ""
                                }`}
                              >
                                <DeferredTextInput
                                  className={`w-full px-1.5 py-1 bg-transparent hover:bg-slate-100 focus:bg-white border border-transparent hover:border-slate-300 focus:border-slate-300 rounded text-[11px] font-mono font-bold focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all truncate text-center ${
                                    hasApproved
                                      ? "text-emerald-800"
                                      : "text-slate-705"
                                  }`}
                                  placeholder="M/D/YY"
                                  initialValue={p.approvedDate || ""}
                                  disabled={isDeleted}
                                  onCommit={(val) =>
                                    handleFieldUpdate(p.id, {
                                      approvedDate: val,
                                      isCompleted:
                                        typeof val === "string" &&
                                        val.trim() !== "",
                                    })
                                  }
                                />
                              </td>
                            );

                          case "actions":
                            return (
                              <td
                                key="actions"
                                className={`${r20} ${c2} text-center`}
                              >
                                <div className="flex justify-center items-center gap-1">
                                  {isDeleted ? (
                                    <button
                                      onClick={() => onRestore(p.id)}
                                      className="px-2 py-0.5 bg-emerald-55 hover:bg-emerald-100 border border-emerald-300 text-emerald-700 rounded font-black text-[9px] uppercase tracking-wider cursor-pointer transition-colors shadow-sm"
                                    >
                                      Restore
                                    </button>
                                  ) : (
                                    <>
                                      <button
                                        onClick={() => onEdit(p)}
                                        className="px-1.5 py-0.5 text-blue-600 font-bold hover:underline bg-transparent border-0 cursor-pointer text-[11px]"
                                        title="Open detailed timeline & edit fields"
                                      >
                                        Edit
                                      </button>
                                      <button
                                        onClick={() => onDelete(p.id)}
                                        className="text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded transition-colors cursor-pointer bg-transparent border-0 p-0.5"
                                        title="Delete program review"
                                      >
                                        <Trash2 className="w-3.5 h-3.5" />
                                      </button>
                                    </>
                                  )}
                                </div>
                              </td>
                            );

                          default:
                            return null;
                        }
                      })}
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
