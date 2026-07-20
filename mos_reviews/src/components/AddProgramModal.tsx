/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { ProgramReview } from '../types';
import { recalculateDates } from '../utils/dateHelpers';
import { X, Search, Check, RefreshCw, AlertCircle, FilePlus, Calendar } from 'lucide-react';

interface AddProgramModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAddPrograms: (newPrograms: ProgramReview[]) => void;
  programs: ProgramReview[];
}

export default function AddProgramModal({
  isOpen,
  onClose,
  onAddPrograms,
  programs,
}: AddProgramModalProps) {
  const [programNum, setProgramNum] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorText, setErrorText] = useState('');
  const [instruments, setInstruments] = useState<Record<string, boolean>>({
    NIRCam: false,
    NIRSpec: true,
    NIRISS: false,
    MIRI: false,
  });
  const [successPrograms, setSuccessPrograms] = useState<Array<{
    program: string;
    cycle: string;
    pi: string;
    aptPrep: string;
    nirspecReviewer?: string;
    nircamReviewer?: string;
    miriReviewer?: string;
    nirissReviewer?: string;
    programInfoUrl: string;
    visitStatusUrl: string;
    results: Array<{
      observation: string;
      obsEarliest: string;
      obsLatest: string;
    }>;
  }>>([]);

  // Reset state on open/close
  useEffect(() => {
    if (isOpen) {
      setProgramNum('');
      setErrorText('');
      setSuccessPrograms([]);
      setLoading(false);
      setInstruments({
        NIRCam: false,
        NIRSpec: true,
        NIRISS: false,
        MIRI: false,
      });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleFetchInfo = async () => {
    const ids = programNum
      .split(/[,\s;|\/:\-+]+/)
      .map((id) => id.trim())
      .filter((id) => id.length > 0 && /^\d+$/.test(id));

    if (ids.length === 0) {
      setErrorText('Please enter one or more valid program numbers separated by commas, spaces, or other punctuation.');
      return;
    }

    // Check if any entered ID already exists in the system (excluding deleted programs)
    const existingIds = new Set(programs.filter(p => !p.isDeleted).map((p) => p.program.trim()));
    const alreadyExisting = ids.filter((id) => existingIds.has(id));

    if (alreadyExisting.length > 0) {
      setErrorText(`Program ID(s) ${alreadyExisting.join(', ')} already exist in your active table. To avoid duplicates, please delete the existing entries first if you wish to re-import.`);
      return;
    }

    setLoading(true);
    setErrorText('');
    setSuccessPrograms([]);

    const activeInstruments = Object.entries(instruments)
      .filter(([_, checked]) => checked)
      .map(([name]) => name)
      .join(',');

    const successfulFetched: typeof successPrograms = [];
    const failedIds: string[] = [];

    try {
      await Promise.all(
        ids.map(async (id) => {
          try {
            const res = await fetch(`/api/parse-stsci?program=${id}&instruments=${encodeURIComponent(activeInstruments)}`);
            if (!res.ok) {
              throw new Error(`Server returned status ${res.status}`);
            }
            const data = await res.json();
            if (!data.success) {
              throw new Error(data.error || 'Failed to fetch program data');
            }

            successfulFetched.push({
              program: data.program,
              cycle: data.cycle || '4',
              pi: data.pi || '',
              aptPrep: data.aptPrep || '',
              nirspecReviewer: data.nirspecReviewer || '',
              nircamReviewer: data.nircamReviewer || '',
              miriReviewer: data.miriReviewer || '',
              nirissReviewer: data.nirissReviewer || '',
              programInfoUrl: data.programInfoUrl || `https://www.stsci.edu/jwst/science-execution/program-information?id=${data.program}`,
              visitStatusUrl: data.visitStatusUrl || `https://www.stsci.edu/jwst-program-info/visits/?program=${data.program}`,
              results: data.results || []
            });
          } catch (err: any) {
            console.error(`Error fetching ID ${id}:`, err);
            failedIds.push(id);
          }
        })
      );

      if (successfulFetched.length === 0) {
        setErrorText(`Failed to retrieve program info for any entered ID (${failedIds.join(', ')}). Please verify the ID numbers or try again.`);
      } else {
        // Sort successful imports numeric ascending for consistency
        successfulFetched.sort((a, b) => parseInt(a.program) - parseInt(b.program));
        setSuccessPrograms(successfulFetched);
        if (failedIds.length > 0) {
          setErrorText(`Successfully retrieved ${successfulFetched.length} program(s), but failed to fetch some IDs: ${failedIds.join(', ')}.`);
        }
      }
    } catch (err: any) {
      console.error(err);
      setErrorText(`An unexpected error occurred during bulk fetch: ${err.message}.`);
    } finally {
      setLoading(false);
    }
  };

  const handleInsert = () => {
    if (successPrograms.length === 0) return;

    // Build program review items for each observation/plan window found
    const recordsToEmit: ProgramReview[] = [];
    const timestamp = Date.now();

    successPrograms.forEach((prog, pIdx) => {
      if (prog.results.length === 0) {
        // Create a single record with empty plan windows if none found
        const calcs = recalculateDates('');
        recordsToEmit.push({
          id: `prog-${prog.program}-obs-all-${timestamp}-${pIdx}`,
          cycle: prog.cycle,
          pi: prog.pi || 'Unassigned',
          aptPrep: prog.aptPrep || '',
          program: prog.program,
          observation: '',
          programInfoUrl: prog.programInfoUrl,
          visitStatusUrl: prog.visitStatusUrl,
          offset: '56',
          submissionDueDate: calcs.submissionDueDate,
          submittedDate: '',
          reviewDueDate: calcs.reviewDueDate,
          reviewedDate: '',
          finalizeDueDate: calcs.finalizeDueDate,
          approvedDate: '',
          sparDeadline: '',
          flightReadyEarliest: calcs.flightReadyEarliest,
          flightReadyMid: calcs.flightReadyMid,
          flightReadyLatest: calcs.flightReadyLatest,
          obsEarliest: '',
          obsMid: '',
          obsLatest: '',
          isCompleted: false,
          notes: '',
          nirspecReviewer: prog.nirspecReviewer || '',
          nircamReviewer: prog.nircamReviewer || '',
          miriReviewer: prog.miriReviewer || '',
          nirissReviewer: prog.nirissReviewer || '',
        });
      } else {
        prog.results.forEach((obs, idx) => {
          const calcs = recalculateDates(obs.obsEarliest);
          const uniqueId = `prog-${prog.program}-obs-${obs.observation || 'all'}-${idx}-${timestamp}-${pIdx}`;
          recordsToEmit.push({
            id: uniqueId,
            cycle: prog.cycle,
            pi: prog.pi || 'Unassigned',
            aptPrep: prog.aptPrep || '',
            program: prog.program,
            observation: obs.observation,
            programInfoUrl: prog.programInfoUrl,
            visitStatusUrl: prog.visitStatusUrl,
            offset: '56',
            submissionDueDate: calcs.submissionDueDate,
            submittedDate: '',
            reviewDueDate: calcs.reviewDueDate,
            reviewedDate: '',
            finalizeDueDate: calcs.finalizeDueDate,
            approvedDate: '',
            sparDeadline: '',
            flightReadyEarliest: calcs.flightReadyEarliest,
            flightReadyMid: calcs.flightReadyMid,
            flightReadyLatest: calcs.flightReadyLatest,
            obsEarliest: obs.obsEarliest,
            obsMid: obs.obsEarliest ? obs.obsEarliest : '', // fallback
            obsLatest: obs.obsLatest,
            isCompleted: false,
            notes: '',
            nirspecReviewer: prog.nirspecReviewer || '',
            nircamReviewer: prog.nircamReviewer || '',
            miriReviewer: prog.miriReviewer || '',
            nirissReviewer: prog.nirissReviewer || '',
          });
        });
      }
    });

    onAddPrograms(recordsToEmit);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4 z-50 overflow-y-auto">
      <div className="bg-white rounded border border-slate-300 shadow-xl w-full max-w-md overflow-hidden flex flex-col scale-100 transition-all duration-300">
        
        {/* Header */}
        <div className="bg-slate-50 border-b border-slate-200 px-4 py-3 flex items-center justify-between">
          <div>
            <h2 className="text-xs sm:text-sm font-black text-slate-950 uppercase tracking-wider flex items-center gap-1.5">
              <FilePlus className="w-4 h-4 text-blue-600" />
              <span>Add JWST Program(s)</span>
            </h2>
            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wide">
              Scrape and import review cycles directly from STScI logs.
            </p>
          </div>
          <button 
            onClick={onClose} 
            className="text-slate-400 hover:text-slate-600 p-1 hover:bg-slate-100 rounded transition-colors cursor-pointer border-0 bg-transparent animate-none"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4 text-xs">
          {errorText && (
            <div className="bg-rose-50 border border-rose-200 text-rose-700 p-2.5 rounded font-semibold text-[11px] flex items-start gap-1.5">
              <AlertCircle className="w-4 h-4 text-rose-500 shrink-0 relative top-0.5" />
              <div className="flex-1">
                <span>{errorText}</span>
              </div>
            </div>
          )}

          {(!successPrograms || successPrograms.length === 0) ? (
            <div className="space-y-3">
              <div>
                <label className="block text-[10px] uppercase font-bold text-slate-500 mb-1">Enter JWST Program ID(s) (separated by spaces, commas, etc.) <span className="text-rose-500">*</span></label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <span className="absolute inset-y-0 left-0 flex items-center pl-2.5 text-slate-400">
                       <Search className="w-3.5 h-3.5" />
                    </span>
                    <input
                      type="text"
                      className="w-full pl-8 pr-2.5 py-1.5 bg-white border border-slate-300 rounded text-xs focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono tracking-wider"
                      placeholder="e.g. 10264 10265; 10266"
                      value={programNum}
                      onChange={(e) => setProgramNum(e.target.value.replace(/[^0-9\s,;|/:\-+]/g, ''))}
                      disabled={loading}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleFetchInfo();
                        }
                      }}
                    />
                  </div>
                  <button
                    type="button"
                    onClick={handleFetchInfo}
                    disabled={loading || !programNum.trim()}
                    className="bg-blue-600 font-bold hover:bg-blue-700 active:bg-blue-800 disabled:opacity-40 text-white text-xs px-4 py-1.5 rounded transition-colors flex items-center gap-1.5 cursor-pointer border-0 shadow-xs"
                  >
                    {loading && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                    <span>{loading ? 'Retrieving...' : 'Fetch'}</span>
                  </button>
                </div>
              </div>

              <div>
                <span className="block text-[10px] uppercase font-bold text-slate-500 mb-1.5">Retrieve Instrument(s):</span>
                <div className="grid grid-cols-2 gap-2 bg-slate-50 border border-slate-200 rounded p-2.5">
                  {['NIRCam', 'NIRSpec', 'NIRISS', 'MIRI'].map((inst) => (
                    <label key={inst} className="flex items-center gap-2 text-slate-700 font-extrabold text-[11px] cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={instruments[inst]}
                        disabled={loading}
                        onChange={(e) => {
                          setInstruments(prev => ({
                            ...prev,
                            [inst]: e.target.checked
                          }));
                        }}
                        className="w-3.5 h-3.5 text-blue-600 border-slate-300 rounded focus:ring-blue-500 cursor-pointer"
                      />
                      <span>{inst}</span>
                    </label>
                  ))}
                </div>
              </div>

              <p className="text-[10px] text-slate-500 leading-normal font-medium max-w-[95%]">
                Enter single or multiple program numbers separated by commas, spaces, semicolons, dashes, or slashes. We fetch their metadata from STScI, extracting the Cycle, PI name, and all Scheduling Windows per Observation.
              </p>
            </div>
          ) : (
            <div className="space-y-3.5 max-h-[350px] overflow-y-auto pr-1">
              <div className="flex items-center gap-1.5 text-emerald-800 font-bold text-xs uppercase">
                <Check className="w-4 h-4 bg-emerald-100 rounded-full p-0.5 animate-bounce" />
                <span>Found Program Information! ({successPrograms.length})</span>
              </div>
              
              {successPrograms.map((prog, pIdx) => (
                <div key={prog.program} className="bg-emerald-55/5 bg-emerald-50/45 border border-emerald-250/50 rounded-md p-3 text-emerald-950 space-y-2">
                  <div className="flex items-center justify-between border-b border-emerald-100/70 pb-1">
                    <span className="font-bold text-slate-900 text-xs font-mono">Program #{prog.program}</span>
                    <span className="text-[10px] bg-emerald-100/80 px-1.5 py-0.5 rounded text-emerald-800 font-bold">Cycle {prog.cycle}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-y-1.5 gap-x-2 text-[11px] font-semibold">
                    <div className="col-span-2">
                      <span className="text-slate-500 uppercase text-[9px] block">Principal Investigator (PI):</span>
                      <span className="font-bold text-slate-900">{prog.pi || '(Unknown)'}</span>
                    </div>
                    {prog.aptPrep && (
                      <div className="col-span-2">
                        <span className="text-slate-500 uppercase text-[9px] block">APT Prep (Reviewer):</span>
                        <span className="font-bold text-slate-800">{prog.aptPrep}</span>
                      </div>
                    )}
                    {(prog.nirspecReviewer || prog.nircamReviewer || prog.miriReviewer || prog.nirissReviewer) && (
                      <div className="col-span-2">
                        <span className="text-slate-500 uppercase text-[9px] block">Assignees / Reviewers:</span>
                        <div className="flex flex-wrap gap-1 mt-0.5">
                          {prog.nirspecReviewer && <span className="text-[9px] bg-slate-150 bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">NIRSpec: {prog.nirspecReviewer}</span>}
                          {prog.nircamReviewer && <span className="text-[9px] bg-slate-150 bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">NIRCam: {prog.nircamReviewer}</span>}
                          {prog.miriReviewer && <span className="text-[9px] bg-slate-150 bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">MIRI: {prog.miriReviewer}</span>}
                          {prog.nirissReviewer && <span className="text-[9px] bg-slate-150 bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">NIRISS: {prog.nirissReviewer}</span>}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-2 text-xs">
                    <span className="block text-[10px] uppercase font-bold text-slate-500 mb-1">Parsed Observation Plan Windows ({prog.results.length})</span>
                    {prog.results.length === 0 ? (
                      <p className="font-medium text-amber-700 bg-amber-50/50 p-2 border border-amber-200/50 rounded text-[10px]">
                        No active scheduling windows were found. A placeholder row will be inserted.
                      </p>
                    ) : (
                      <div className="border border-slate-200 rounded max-h-24 overflow-y-auto divide-y divide-slate-100 bg-white">
                        {prog.results.map((item, idx) => (
                          <div key={idx} className="flex justify-between items-center py-1 px-2 font-mono text-[10px]">
                            <span className="font-bold text-blue-700">Obs {item.observation}</span>
                            <span className="font-semibold text-slate-700">{item.obsEarliest} - {item.obsLatest}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-slate-50 border-t border-slate-200 px-4 py-3 flex items-center justify-end gap-2.5">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 text-xs font-bold text-slate-600 bg-white border border-slate-300 rounded hover:bg-slate-50 cursor-pointer transition-colors border-solid"
          >
            Cancel
          </button>
          {successPrograms && successPrograms.length > 0 && (
            <button
               type="button"
               onClick={handleInsert}
               className="bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white font-bold text-xs px-4 py-1.5 rounded cursor-pointer border-0 shadow-sm"
            >
              Add Program and Save
            </button>
          )}
        </div>

      </div>
    </div>
  );
}
