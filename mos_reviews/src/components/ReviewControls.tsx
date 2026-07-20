/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { CompletionFilter, ColumnConfig, DEFAULT_COLUMNS } from '../types';
import { Plus, SlidersHorizontal, Trash2 } from 'lucide-react';

interface ReviewControlsProps {
  completionFilter: CompletionFilter;
  setCompletionFilter: (f: CompletionFilter) => void;
  onAddProgram: () => void;
  reviewerFilterEnabled: boolean;
  setReviewerFilterEnabled: (b: boolean) => void;
  reviewerName: string;
  setReviewerName: (s: string) => void;
  columns: ColumnConfig[];
  setColumns: React.Dispatch<React.SetStateAction<ColumnConfig[]>>;
  duplicateCount: number;
  onRemoveDuplicates: () => void;
  rowPitch: number;
  setRowPitch: (n: number) => void;
  colPitch: number;
  setColPitch: (n: number) => void;
}

export default function ReviewControls({
  completionFilter,
  setCompletionFilter,
  onAddProgram,
  reviewerFilterEnabled,
  setReviewerFilterEnabled,
  reviewerName,
  setReviewerName,
  columns,
  setColumns,
  duplicateCount,
  onRemoveDuplicates,
  rowPitch,
  setRowPitch,
  colPitch,
  setColPitch,
}: ReviewControlsProps) {

  const [showConfig, setShowConfig] = useState(false);
  const visibleColumnsCount = columns.filter(c => c.visible).length;

  const resetColumns = () => {
    setColumns(DEFAULT_COLUMNS);
  };

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 sm:p-3.5 mb-4 shadow-2xs">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        
        {/* Left Side: Add Program Button and Duplicates Actions */}
        <div className="flex items-center gap-2 justify-start md:w-1/4">
          <button
            id="btn-add-program"
            onClick={onAddProgram}
            className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white px-3 py-1.5 rounded text-xs font-bold cursor-pointer transition-colors shadow-2xs whitespace-nowrap"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add Program(s)</span>
          </button>

          {duplicateCount > 0 && (
            <button
              id="btn-remove-duplicates"
              onClick={onRemoveDuplicates}
              className="flex items-center gap-1.5 bg-amber-50 hover:bg-amber-100 active:bg-amber-150 border border-amber-300 text-amber-800 px-2.5 py-1.5 rounded text-xs font-semibold cursor-pointer transition-colors shadow-2xs whitespace-nowrap"
              title="Click to remove duplicate program observation rows"
            >
              <Trash2 className="w-3.5 h-3.5 text-amber-600" />
              <span>Deduplicate ({duplicateCount})</span>
            </button>
          )}
        </div>

        {/* Center Side: View Filters and Assigned Reviewer Dialg */}
        <div className="flex flex-wrap items-center justify-start md:justify-center gap-3 md:flex-1">
          
          {/* Completion Status Filter */}
          <div className="flex items-center gap-1.5 text-xs font-semibold">
            <span className="text-slate-500 text-[10px] sm:text-[11px] uppercase tracking-wider font-bold">View:</span>
            <div className="flex bg-slate-200/60 p-0.5 rounded">
              <button
                id="filter-all"
                className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all cursor-pointer ${completionFilter === 'all' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
                onClick={() => setCompletionFilter('all')}
              >
                All
              </button>
              <button
                id="filter-pending"
                className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all cursor-pointer ${completionFilter === 'pending' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
                onClick={() => setCompletionFilter('pending')}
              >
                Pending
              </button>
              <button
                id="filter-completed"
                className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all cursor-pointer ${completionFilter === 'completed' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
                onClick={() => setCompletionFilter('completed')}
              >
                Completed
              </button>
              <button
                id="filter-deleted"
                className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all cursor-pointer ${completionFilter === 'deleted' ? 'bg-rose-105 text-rose-700 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
                onClick={() => setCompletionFilter('deleted')}
              >
                Deleted
              </button>
            </div>
          </div>

          <div className="h-4 w-[1px] bg-slate-300 hidden sm:block"></div>

          {/* Assigned Reviewer Filter with Editable Name */}
          <div className="flex items-center gap-1.5 text-xs font-semibold bg-indigo-50/50 border border-indigo-200/60 p-1 rounded-md shadow-2xs">
            <button
              onClick={() => setReviewerFilterEnabled(!reviewerFilterEnabled)}
              className={`px-2 py-1 text-[11px] font-bold rounded cursor-pointer transition-all ${
                reviewerFilterEnabled
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-white text-indigo-700 hover:bg-indigo-50 border border-indigo-200/70'
              }`}
              title="Show only programs with the specified name in reviewer or analyst fields"
            >
              {reviewerFilterEnabled ? 'Showing Assigned Only' : 'Filter Assigned To:'}
            </button>
            <input
              type="text"
              placeholder="e.g. Dan Coe"
              className="w-24 px-1.5 py-0.5 bg-white border border-slate-300 rounded text-[11px] focus:outline-none focus:ring-1 focus:ring-indigo-550 font-semibold text-slate-705 shadow-sm"
              value={reviewerName}
              onChange={(e) => setReviewerName(e.target.value)}
            />
          </div>
        </div>

        {/* Right Side: Manage Columns Button and Menu */}
        <div className="flex justify-start md:justify-end md:w-1/4 relative">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 bg-white hover:bg-slate-50 border border-slate-300 hover:border-slate-400 text-slate-700 hover:text-slate-900 rounded-md shadow-2xs text-[11px] font-extrabold cursor-pointer transition-all h-[30px]"
            title="Show, hide, or reset table columns visibility"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-500" />
            <span>Manage Columns ({visibleColumnsCount})</span>
          </button>

          {showConfig && (
            <div className="absolute right-0 mt-8 w-56 bg-white border border-slate-350 rounded-lg shadow-xl p-3 z-50 space-y-2 text-xs">
              <div className="flex justify-between items-center border-b border-slate-200 pb-1.5 mb-1.5">
                <span className="font-bold text-slate-800 text-[10px] uppercase tracking-wider">Column Visibility</span>
                <button
                  onClick={resetColumns}
                  className="text-blue-600 hover:underline hover:text-blue-700 bg-transparent border-0 cursor-pointer font-extrabold text-[9px] uppercase tracking-wide"
                >
                  Reset Defaults
                </button>
              </div>
              <div className="max-h-52 overflow-y-auto space-y-1">
                {columns.map((col) => (
                  <label key={col.key} className="flex items-center gap-2 cursor-pointer py-0.5 hover:bg-slate-50 rounded px-1.5 font-bold text-slate-700 select-none">
                    <input
                      type="checkbox"
                      checked={col.visible}
                      onChange={() => {
                        const visibleCount = columns.filter(c => c.visible).length;
                        if (col.visible && visibleCount <= 1) return; // Prevent hiding all columns
                        setColumns(prev => prev.map(c => c.key === col.key ? { ...c, visible: !c.visible } : c));
                      }}
                      className="w-3.5 h-3.5 text-blue-600 border-slate-300 rounded focus:ring-blue-500 cursor-pointer"
                    />
                    <span>{col.label}</span>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

      </div>

      {/* Table Layout Density Settings */}
      <div className="mt-2.5 pt-2.5 border-t border-slate-200/80 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-semibold text-slate-500">
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500 text-[10px] uppercase tracking-wider font-bold">
            Table Layout:
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-6">
          {/* Row Pitch Slider */}
          <div className="flex items-center gap-2">
            <span id="label-row-pitch" className="text-slate-500 text-[11px] font-semibold">Row Pitch:</span>
            <input
              type="range"
              min="0"
              max="20"
              step="1"
              value={rowPitch}
              onChange={(e) => setRowPitch(Number(e.target.value))}
              className="w-24 accent-blue-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              title="Drag to adjust vertical padding between table rows"
            />
            <span className="font-mono text-[11px] bg-slate-200/50 text-slate-650 px-1.5 py-0.5 rounded">
              {rowPitch === 0 ? "0px (Min)" : `${rowPitch}px`}
            </span>
          </div>

          {/* Col Pitch Slider */}
          <div className="flex items-center gap-2">
            <span id="label-col-pitch" className="text-slate-550 text-[11px] font-semibold font-sans">Col Pitch:</span>
            <input
              type="range"
              min="0"
              max="20"
              step="1"
              value={colPitch}
              onChange={(e) => setColPitch(Number(e.target.value))}
              className="w-24 accent-blue-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              title="Drag to adjust horizontal padding between table columns"
            />
            <span className="font-mono text-[11px] bg-slate-200/50 text-slate-650 px-1.5 py-0.5 rounded">
              {colPitch === 0 ? "0px (Min)" : `${colPitch}px`}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
