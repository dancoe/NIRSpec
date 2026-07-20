/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

// Native TS-compatible date operations that avoid library dependencies and render fast.

export function parseSheetDate(dateStr: string): Date | null {
  if (!dateStr || dateStr.includes('#') || dateStr.includes('no')) return null;
  
  // Handles format: "M/D/YY" or "M/D/YYYY" or "Sun  7/19" (where the date is extracted)
  let datePart = dateStr.trim();
  
  // If there's a day of the week prefix like "Sun  7/19", extract the "7/19" part
  const match = datePart.match(/(?:[A-Za-z]+\s+)?(\d+\/\d+\/\d+)/);
  if (match) {
    datePart = match[1];
  } else {
    // Try simple "M/D" and assume year is 2026 (Cycle 4 is mostly 2025-2027)
    const shortMatch = datePart.match(/(?:[A-Za-z]+\s+)?(\d+\/\d+)/);
    if (shortMatch) {
      datePart = `${shortMatch[1]}/26`;
    }
  }

  const parts = datePart.split('/');
  if (parts.length < 2) return null;
  
  let month = parseInt(parts[0], 10) - 1;
  let day = parseInt(parts[1], 10);
  let year = parseInt(parts[2], 10);
  
  if (isNaN(month) || isNaN(day)) return null;
  
  if (isNaN(year)) {
    year = 2026; // fallback for short dates
  } else if (year < 100) {
    // 2-digit year conversion
    year = 2000 + year;
  }
  
  const d = new Date(year, month, day);
  return isNaN(d.getTime()) ? null : d;
}

export function formatSheetDate(date: Date | null): string {
  if (!date) return '';
  const month = date.getMonth() + 1;
  const day = date.getDate();
  const year = date.getFullYear() % 100;
  return `${month}/${day}/${year.toString().padStart(2, '0')}`;
}

export function formatSheetDateWithDay(date: Date | null): string {
  if (!date) return '';
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const dayName = days[date.getDay()];
  const month = date.getMonth() + 1;
  const day = date.getDate();
  const year = date.getFullYear() % 100;
  return `${dayName}  ${month}/${day}/${year.toString().padStart(2, '0')}`;
}

export function addDays(date: Date, days: number): Date {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
}

export function getDaysDifference(d1: Date, d2: Date): number {
  const utc1 = Date.UTC(d1.getFullYear(), d1.getMonth(), d1.getDate());
  const utc2 = Date.UTC(d2.getFullYear(), d2.getMonth(), d2.getDate());
  return Math.floor((utc2 - utc1) / (1000 * 60 * 60 * 24));
}

// Strip day name + extra spaces (e.g. "Sun  7/19/26" -> "7/19/26" or "Sun  7/19" -> "7/19")
export function cleanWeekdayAndFormat(dateStr: string): string {
  if (!dateStr) return '';
  return dateStr.replace(/^[A-Za-z]{3}\s+/, '').trim();
}

// Ensures year and correct weekday abbreviation are appended/formatted
export function formatFlightReadyWithYear(frVal: string, obsEarliestVal: string): string {
  if (!frVal) return '';
  
  // Try to find if frVal already has a weekday prefix
  const weekdayMatch = frVal.match(/^([A-Za-z]{3})\s+/);
  const weekdayPrefix = weekdayMatch ? `${weekdayMatch[1]}  ` : '';
  
  const clean = cleanWeekdayAndFormat(frVal);
  if (!clean) return '';
  
  let formatted = clean;
  const slashes = (clean.match(/\//g) || []).length;
  if (slashes < 2) {
    // Parse year from obsEarliestVal if possible
    let year = '26';
    if (obsEarliestVal) {
      const parts = obsEarliestVal.split('/');
      if (parts.length >= 3) {
        year = parts[2].trim();
      }
    }
    formatted = `${clean}/${year}`;
  }
  
  // If we can parse it as a Date, we get the correct weekday for it!
  const parsed = parseSheetDate(formatted);
  if (parsed) {
    return formatSheetDateWithDay(parsed);
  }
  
  return weekdayPrefix + formatted;
}

// Recalculates all dependent dates for a program review based on its obsEarliest date string
export function recalculateDates(obsEarliestRaw: string): {
  submissionDueDate: string;
  reviewDueDate: string;
  finalizeDueDate: string;
  flightReadyEarliest: string;
  flightReadyMid: string;
  flightReadyLatest: string;
} {
  const obsEarliest = parseSheetDate(obsEarliestRaw);
  if (!obsEarliest) {
    return {
      submissionDueDate: '',
      reviewDueDate: '',
      finalizeDueDate: '',
      flightReadyEarliest: '',
      flightReadyMid: '',
      flightReadyLatest: ''
    };
  }

  // 1. Submission Due Date = Obs Earliest minus 62 days, ending on a Monday (making it 56 – 62 days on a Monday before the plan window start)
  const subDateBase = addDays(obsEarliest, -62);
  const dayOfWeek = subDateBase.getDay(); // 0 = Sunday, 1 = Monday, etc.
  const daysToAdd = (1 - dayOfWeek + 7) % 7;
  const subDate = addDays(subDateBase, daysToAdd);

  // 1.5. Review Due Date = Obs Earliest minus 42 days (6 weeks)
  const revDate = addDays(obsEarliest, -42);
  
  // 2. Finalize Due Date = Obs Earliest minus 28 days
  const finDate = addDays(obsEarliest, -28);
  
  // 3. Flight Ready Earliest = Obs Earliest minus 18 days
  const frEarliest = addDays(obsEarliest, -18);
  
  // 4. Flight Ready Latest = Obs Earliest minus 11 days
  const frLatest = addDays(obsEarliest, -11);
  
  // 5. Flight Ready Mid = The Wednesday between Flight Ready Earliest and Flight Ready Latest
  let frMid: Date | null = null;
  for (let d = new Date(frEarliest); d <= frLatest; d.setDate(d.getDate() + 1)) {
    if (d.getDay() === 3) { // 3 is Wednesday
      frMid = new Date(d);
      break;
    }
  }
  if (!frMid) {
    // fallback midpoint if no Wednesday was found in range (rare edge cases in formatting)
    frMid = addDays(frEarliest, 3);
  }

  return {
    submissionDueDate: formatSheetDateWithDay(subDate),
    reviewDueDate: formatSheetDateWithDay(revDate),
    finalizeDueDate: formatSheetDateWithDay(finDate),
    flightReadyEarliest: formatSheetDateWithDay(frEarliest),
    flightReadyMid: formatSheetDateWithDay(frMid),
    flightReadyLatest: formatSheetDateWithDay(frLatest)
  };
}
