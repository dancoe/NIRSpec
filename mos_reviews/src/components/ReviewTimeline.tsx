/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ProgramReview } from '../types';
import { parseSheetDate, getDaysDifference } from '../utils/dateHelpers';
import { motion } from 'motion/react';
import { Calendar, CheckCircle2, ChevronRight, Clock, AlertCircle } from 'lucide-react';

interface ReviewTimelineProps {
  programs: ProgramReview[];
  referenceDate: Date;
  onEdit: (program: ProgramReview) => void;
  onToggleComplete: (id: string) => void;
}

export default function ReviewTimeline({
  programs,
  referenceDate,
  onEdit,
  onToggleComplete,
}: ReviewTimelineProps) {
  
  // Only display programs that have at least a submission due date or obs earliest to map on timeline
  const timelinePrograms = programs.filter(p => p.submissionDueDate || p.obsEarliest);

  const getTimelineCardStatus = (p: ProgramReview) => {
    if (p.isCompleted) return { border: 'border-l-4 border-l-emerald-500', bg: 'bg-emerald-50/5' };
    
    // Check if overdue
    if (p.submissionDueDate) {
      const subDate = parseSheetDate(p.submissionDueDate);
      if (subDate && getDaysDifference(referenceDate, subDate) < 0 && !p.submittedDate) {
        return { border: 'border-l-4 border-l-rose-500', bg: 'bg-rose-50/10' };
      }
    }
    
    if (p.finalizeDueDate) {
      const finDate = parseSheetDate(p.finalizeDueDate);
      if (finDate && getDaysDifference(referenceDate, finDate) < 0 && !p.approvedDate) {
        return { border: 'border-l-4 border-l-rose-500', bg: 'bg-rose-50/10' };
      }
    }

    // Checking if soon
    if (p.submissionDueDate) {
      const subDate = parseSheetDate(p.submissionDueDate);
      if (subDate) {
        const days = getDaysDifference(referenceDate, subDate);
        if (days >= 0 && days <= 30) {
          return { border: 'border-l-4 border-l-amber-500', bg: 'bg-amber-50/10' };
        }
      }
    }

    return { border: 'border-l-4 border-l-blue-500', bg: 'bg-white' };
  };

  return (
    <div className="space-y-3">
      {timelinePrograms.length === 0 ? (
        <div className="bg-white rounded border border-slate-200 p-8 text-center text-slate-400 font-semibold text-xs">
          No program reviews available to display in chronological timeline.
        </div>
      ) : (
        <div className="relative border-l border-slate-200 ml-3 pl-4 space-y-4">
          {timelinePrograms.map((p, index) => {
            const statusStyle = getTimelineCardStatus(p);
            const subDateObj = parseSheetDate(p.submissionDueDate);
            const finDateObj = parseSheetDate(p.finalizeDueDate);
            const obsDateObj = parseSheetDate(p.obsEarliest);
            
            const daysToSub = subDateObj ? getDaysDifference(referenceDate, subDateObj) : null;
            const daysToFin = finDateObj ? getDaysDifference(referenceDate, finDateObj) : null;

            return (
              <motion.div
                key={p.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, delay: Math.min(index * 0.03, 0.3) }}
                className={`relative rounded border border-slate-200 p-3.5 shadow-xs transition-all ${statusStyle.border} ${statusStyle.bg}`}
              >
                {/* Timeline Dot Marker in Left margin */}
                <div className={`absolute -left-[23px] top-5.5 w-2.5 h-2.5 rounded-full border ${
                  p.isCompleted ? 'bg-emerald-500 border-emerald-500' : 
                  (daysToSub !== null && daysToSub < 0 && !p.submittedDate) ? 'bg-rose-500 border-rose-500 animate-pulse' : 
                  (daysToSub !== null && daysToSub <= 30) ? 'bg-amber-500 border-amber-500' : 'bg-blue-500 border-blue-500'
                }`} />

                {/* Top Row with Header Actions */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 mb-3 border-b border-slate-100 pb-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest bg-slate-150 px-1.5 py-0.5 rounded border border-slate-200">
                      Prog {p.program}
                    </span>
                    {p.observation && (
                      <span className="text-[10px] font-mono font-bold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
                        Obs {p.observation}
                      </span>
                    )}
                    <h3 className="font-bold text-slate-900 text-xs">
                      {p.pi || 'Unassigned Catalog'}
                    </h3>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      id={`timeline-toggle-${p.id}`}
                      onClick={() => onToggleComplete(p.id)}
                      className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded border transition-colors cursor-pointer ${
                        p.isCompleted 
                          ? 'bg-emerald-100 border-emerald-200 text-emerald-800' 
                          : 'bg-white border-slate-300 text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      {p.isCompleted ? 'Completed' : 'Mark Complete'}
                    </button>
                    <button
                      id={`timeline-edit-${p.id}`}
                      onClick={() => onEdit(p)}
                      className="text-[10px] uppercase font-bold bg-blue-50 border border-blue-200 text-blue-600 px-2 py-0.5 rounded hover:bg-blue-100 cursor-pointer"
                    >
                      Update
                    </button>
                  </div>
                </div>

                {/* Grid details */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                  {/* Submission stage */}
                  <div className="bg-slate-50/50 p-2 rounded border border-slate-200/60">
                    <div className="flex items-center justify-between text-[9px] text-slate-500 font-bold uppercase tracking-wider mb-1">
                      <span>Phase I: APT Submission</span>
                      {p.submittedDate && <span className="text-emerald-600 font-bold">Done</span>}
                    </div>
                    <p className="text-xs font-bold text-slate-800 flex items-center gap-1">
                      <Calendar className="w-3.5 h-3.5 text-slate-400" />
                      <span>{p.submissionDueDate || 'No Due Date'}</span>
                    </p>
                    <div className="mt-1 text-[10px]">
                      {p.submittedDate ? (
                        <p className="text-emerald-600 font-bold">Submitted on: {p.submittedDate}</p>
                      ) : daysToSub !== null ? (
                        daysToSub < 0 ? (
                          <span className="text-rose-600 font-black flex items-center gap-1">
                            Overdue by {Math.abs(daysToSub)} days
                          </span>
                        ) : (
                          <span className="text-amber-600 font-bold flex items-center gap-1">
                            Due in {daysToSub} days
                          </span>
                        )
                      ) : (
                        <p className="text-slate-400 italic">No sub countdown</p>
                      )}
                    </div>
                  </div>

                  {/* Finalization stage */}
                  <div className="bg-slate-50/50 p-2 rounded border border-slate-200/60">
                    <div className="flex items-center justify-between text-[9px] text-slate-500 font-bold uppercase tracking-wider mb-1">
                      <span>Phase II: APT Finalize</span>
                      {p.approvedDate && <span className="text-emerald-600 font-bold">Approved</span>}
                    </div>
                    <p className="text-xs font-bold text-slate-800 flex items-center gap-1">
                      <Calendar className="w-3.5 h-3.5 text-slate-400" />
                      <span>{p.finalizeDueDate || 'No Due Date'}</span>
                    </p>
                    <div className="mt-1 text-[10px]">
                      {p.approvedDate ? (
                        <p className="text-emerald-600 font-bold">Approved on: {p.approvedDate}</p>
                      ) : daysToFin !== null ? (
                        daysToFin < 0 ? (
                          <span className="text-rose-600 font-black flex items-center gap-1">
                            Overdue by {Math.abs(daysToFin)} days
                          </span>
                        ) : (
                          <span className="text-blue-600 font-bold flex items-center gap-1">
                            Due in {daysToFin} days
                          </span>
                        )
                      ) : (
                        <p className="text-slate-400 italic">No finalize countdown</p>
                      )}
                    </div>
                  </div>

                  {/* Flight Ready Stage */}
                  <div className="bg-slate-50/50 p-2 rounded border border-slate-200/60">
                    <div className="text-[9px] text-slate-500 font-bold uppercase tracking-wider mb-1">
                      <span>Phase III: Flight Ready</span>
                    </div>
                    <div className="text-[10px] text-slate-600 space-y-0.5 mt-0.5">
                      {p.flightReadyEarliest && (
                        <p className="flex justify-between font-medium">
                          <span className="text-slate-400">Earliest SPAR:</span>
                          <span className="font-mono text-[10px] font-bold">{p.flightReadyEarliest}</span>
                        </p>
                      )}
                      {p.flightReadyMid && (
                        <p className="flex justify-between font-medium">
                          <span className="text-slate-400">Mid Flight:</span>
                          <span className="font-mono text-[10px] font-bold">{p.flightReadyMid}</span>
                        </p>
                      )}
                      {p.flightReadyLatest && (
                        <p className="flex justify-between font-medium">
                          <span className="text-slate-400">Latest Flight:</span>
                          <span className="font-mono text-[10px] font-bold">{p.flightReadyLatest}</span>
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Milestone Progress Path */}
                <div className="flex flex-wrap items-center justify-between bg-slate-50 p-2 rounded text-[10px] font-bold gap-1.5 border border-slate-200">
                  <div className="flex items-center gap-1 min-w-[110px]">
                    <div className={`w-2.5 h-2.5 rounded-full ${p.isCompleted || p.submittedDate ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                    <span className={p.submittedDate ? 'text-emerald-700' : 'text-slate-500'}>APT Submitted</span>
                  </div>
                  <ChevronRight className="w-3.5 h-3.5 text-slate-300 hidden md:block" />

                  <div className="flex items-center gap-1 min-w-[110px]">
                    <div className={`w-2.5 h-2.5 rounded-full ${p.isCompleted || p.reviewedDate ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                    <span className={p.reviewedDate ? 'text-emerald-700' : 'text-slate-500'}>Reviewed</span>
                  </div>
                  <ChevronRight className="w-3.5 h-3.5 text-slate-300 hidden md:block" />

                  <div className="flex items-center gap-1 min-w-[110px]">
                    <div className={`w-2.5 h-2.5 rounded-full ${p.isCompleted || p.approvedDate ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                    <span className={p.approvedDate ? 'text-emerald-700' : 'text-slate-500'}>Approved</span>
                  </div>
                  <ChevronRight className="w-3.5 h-3.5 text-slate-300 hidden md:block" />

                  <div className="flex items-center gap-1 min-w-[110px]">
                    <div className={`w-2.5 h-2.5 rounded-full ${p.isCompleted ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                    <span className={p.isCompleted ? 'text-emerald-700' : 'text-slate-500'}>All Closed 🚀</span>
                  </div>
                </div>

                {/* Observational target Info */}
                {p.obsEarliest && (
                  <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400 border-t border-slate-100 pt-1.5">
                    <span>APT Prep: <strong className="text-slate-600 font-bold">{p.aptPrep || 'N/A'}</strong></span>
                    <span>Obs Window: <strong className="text-slate-600 font-bold">{p.obsEarliest}</strong> to <strong className="text-slate-600 font-bold">{p.obsLatest}</strong> (Mid: {p.obsMid})</span>
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
