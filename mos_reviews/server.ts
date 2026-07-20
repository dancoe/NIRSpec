import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";

async function startServer() {
  const app = express();
  const PORT = 5187;

  app.use(express.json());

  // API router/route for fetching and parsing STScI visit pages
  app.get("/api/parse-stsci", async (req, res) => {
    const { program } = req.query;
    if (!program) {
      return res.status(400).json({ error: "Program number is required" });
    }

    try {
      // 1. Fetch of the program info page to get Cycle, PI, and Specific instrument reviewers
      let cycle = "4";
      let pi = "";
      let aptPrep = "";
      let nirspecReviewer = "";
      let nircamReviewer = "";
      let miriReviewer = "";
      let nirissReviewer = "";

      try {
        const infoUrl = `https://www.stsci.edu/jwst-program-info/program/?program=${program}`;
        const infoResponse = await fetch(infoUrl, {
          headers: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          },
        });
        if (infoResponse.ok) {
          const infoHtml = await infoResponse.text();
          const infoText = infoHtml
            .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, " ")
            .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, " ")
            .replace(/<[^>]+>/g, " ")
            .replace(/\s+/g, " ");

          const piMatch = infoText.match(/Principal\s+Investigator:\s*([^P\r\n\t]+?)(?=\s*(?:PI\s+Institution:|Institution:|Investigators:|\[|\r|\n|$))/i);
          if (piMatch) {
            pi = piMatch[1].trim();
          }

          const cycleMatch = infoText.match(/Cycle:\s*(\d+)/i);
          if (cycleMatch) {
            cycle = cycleMatch[1].trim();
          }

          const reviewerMatch = infoText.match(/(?:NIRSPEC\s+Reviewer|Reviewer|APT\s+Prep):\s*([^C\r\n\t\[]+?)(?=\s*(?:\[|Contact:|\r|\n|$))/i);
          if (reviewerMatch) {
            aptPrep = reviewerMatch[1].trim();
          }

          // Specific instrument reviewer lookups in collapsed-space text
          const extractReviewer = (text: string, instrument: string) => {
            const regex = new RegExp(`${instrument}\\s+Reviewer:\\s*([^:]+?)(?=\\s+(?:[A-Z][A-Za-z0-9]+\\s+Reviewer|Contact|Principal|PI|Institution|Co-Investigators|\\[|$))`, 'i');
            const match = text.match(regex);
            return match ? match[1].trim() : "";
          };

          nirspecReviewer = extractReviewer(infoText, "NIRSPEC");
          nircamReviewer = extractReviewer(infoText, "NIRCam");
          miriReviewer = extractReviewer(infoText, "MIRI");
          nirissReviewer = extractReviewer(infoText, "NIRISS");
        }
      } catch (err) {
        console.error(`[API] Error scraping program info page for ${program}:`, err);
      }

      // 2. Fetch the visits page to get plan windows
      let results: Array<{
        observation: string;
        obsEarliest: string;
        obsLatest: string;
      }> = [];

      try {
        const url = `https://www.stsci.edu/jwst-program-info/visits/?program=${program}`;
        console.log(`[API] Fetching STScI visits page: ${url}`);
        
        const response = await fetch(url, {
          headers: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          },
        });

        if (response.ok) {
          const html = await response.text();

          // Clean HTML tags helper to get pure text content of cells
          const cleanText = (htmlCell: string) => {
            return htmlCell
              .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, " ")
              .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, " ")
              .replace(/<[^>]+>/g, " ")
              .replace(/\s+/g, " ")
              .trim();
          };

          const rawResults: Array<{
            observation: string;
            obsEarliest: string;
            obsLatest: string;
            template: string;
          }> = [];

          const trRegex = /<tr[^>]*>([\s\S]*?)<\/tr>/gi;
          const tdRegex = /<td[^>]*>([\s\S]*?)<\/td>/gi;

          let trMatch;
          while ((trMatch = trRegex.exec(html)) !== null) {
            const trContent = trMatch[1];
            const tds: string[] = [];
            let tdMatch;
            while ((tdMatch = tdRegex.exec(trContent)) !== null) {
              tds.push(tdMatch[1]);
            }

            if (tds.length >= 7) {
              const obsNum = cleanText(tds[0]);
              const template = cleanText(tds[4]);
              const planWindows = cleanText(tds[6]);

              // Check if it's a valid row with an observation number
              if (obsNum && !isNaN(parseInt(obsNum, 10)) && planWindows) {
                let earliest = "";
                let latest = "";

                // 1. Check decimal year range first
                const decMatch = planWindows.match(/(20\d{2})\.(\d{3})\s*-\s*(20\d{2})\.(\d{3})/i);
                if (decMatch) {
                  earliest = convertDecimalToDate(parseInt(decMatch[1], 10), parseInt(decMatch[2], 10));
                  latest = convertDecimalToDate(parseInt(decMatch[3], 10), parseInt(decMatch[4], 10));
                } else {
                  // 2. Check date range
                  const dateRangeMatch = planWindows.match(/([A-Z][a-z]{2,8}\s+\d{1,2},\s*\d{4})\s*-\s*([A-Z][a-z]{2,8}\s+\d{1,2},\s*\d{4})/i);
                  if (dateRangeMatch) {
                    earliest = formatDateStr(new Date(dateRangeMatch[1]));
                    latest = formatDateStr(new Date(dateRangeMatch[2]));
                  }
                }

                if (obsNum && earliest && latest) {
                  rawResults.push({
                    observation: obsNum,
                    obsEarliest: earliest,
                    obsLatest: latest,
                    template: template
                  });
                }
              }
            }
          }

          // Fallback text parser when no HTML table rows matched
          if (rawResults.length === 0) {
            const textContent = html
              .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, " ")
              .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, " ")
              .replace(/<[^>]+>/g, " ")
              .replace(/\s+/g, " ");

            const keyword = "Plan Windows for Observation";
            let pos = 0;
            while (true) {
              const index = textContent.indexOf(keyword, pos);
              if (index === -1) break;

              const block = textContent.substring(index, index + 300);
              const obsNumMatch = block.match(/Plan\s+Windows\s+for\s+Observation\s+(\d+)/i);
              if (obsNumMatch) {
                const obsNum = obsNumMatch[1];
                let earliest = "";
                let latest = "";

                const decMatch = block.match(/(20\d{2})\.(\d{3})\s*-\s*(20\d{2})\.(\d{3})/i);
                if (decMatch) {
                  earliest = convertDecimalToDate(parseInt(decMatch[1], 10), parseInt(decMatch[2], 10));
                  latest = convertDecimalToDate(parseInt(decMatch[3], 10), parseInt(decMatch[4], 10));
                } else {
                  const dateRangeMatch = block.match(/([A-Z][a-z]{2,8}\s+\d{1,2},\s*\d{4})\s*-\s*([A-Z][a-z]{2,8}\s+\d{1,2},\s*\d{4})/i);
                  if (dateRangeMatch) {
                    earliest = formatDateStr(new Date(dateRangeMatch[1]));
                    latest = formatDateStr(new Date(dateRangeMatch[2]));
                  }
                }

                if (obsNum && earliest && latest) {
                  rawResults.push({
                    observation: obsNum,
                    obsEarliest: earliest,
                    obsLatest: latest,
                    template: "Unknown"
                  });
                }
              }
              pos = index + keyword.length;
            }

            if (rawResults.length === 0) {
              const decRegex = /(?:Observation\s+(\d+)[^0-9]*)?(20\d{2})\.(\d{3})\s*-\s*(20\d{2})\.(\d{3})/gi;
              let match;
              let idx = 1;
              while ((match = decRegex.exec(textContent)) !== null) {
                const obsNum = match[1] || String(idx++);
                const startYear = parseInt(match[2], 10);
                const startDay = parseInt(match[3], 10);
                const endYear = parseInt(match[4], 10);
                const endDay = parseInt(match[5], 10);

                const earliest = convertDecimalToDate(startYear, startDay);
                const latest = convertDecimalToDate(endYear, endDay);

                rawResults.push({
                  observation: obsNum,
                  obsEarliest: earliest,
                  obsLatest: latest,
                  template: "Unknown"
                });
              }
            }
          }

          // Filter by insts (checked instruments, e.g. NIRSpec, NIRCam) passed via query string
          const allowedInstruments = (req.query.instruments as string || "")
            .split(",")
            .map(i => i.trim().toLowerCase())
            .filter(Boolean);

          const filteredResults = rawResults.filter(r => {
            if (allowedInstruments.length === 0) return true;
            return allowedInstruments.some(inst => r.template.toLowerCase().includes(inst));
          });

          // Group by Observation to avoid duplicate rows
          const groupedResults: Record<string, {
            observation: string;
            obsEarliest: string;
            obsLatest: string;
            earliestDate: Date | null;
            latestDate: Date | null;
          }> = {};

          for (const r of filteredResults) {
            const obs = r.observation;
            const earliest = r.obsEarliest;
            const latest = r.obsLatest;
            
            const parseD = (dStr: string) => {
              if (!dStr) return null;
              const d = new Date(dStr);
              return isNaN(d.getTime()) ? null : d;
            };
            
            const ed = parseD(earliest);
            const ld = parseD(latest);
            
            if (!groupedResults[obs]) {
              groupedResults[obs] = {
                observation: obs,
                obsEarliest: earliest,
                obsLatest: latest,
                earliestDate: ed,
                latestDate: ld
              };
            } else {
              const gr = groupedResults[obs];
              if (ed && (!gr.earliestDate || ed < gr.earliestDate)) {
                gr.earliestDate = ed;
                gr.obsEarliest = earliest;
              }
              if (ld && (!gr.latestDate || ld > gr.latestDate)) {
                gr.latestDate = ld;
                gr.obsLatest = latest;
              }
            }
          }

          results = Object.values(groupedResults).map(gr => ({
            observation: gr.observation,
            obsEarliest: gr.obsEarliest,
            obsLatest: gr.obsLatest
          }));
        }
      } catch (err) {
        console.error(`[API] Error scraping program visits page for ${program}:`, err);
      }

      console.log(`[API] Scraped ${results.length} visits results, cycle ${cycle}, pi ${pi}, analyst ${aptPrep}, NIRSpec Reviewer ${nirspecReviewer} for program ${program}`);
      return res.json({
        success: true,
        program,
        cycle,
        pi,
        aptPrep,
        nirspecReviewer,
        nircamReviewer,
        miriReviewer,
        nirissReviewer,
        programInfoUrl: `https://www.stsci.edu/jwst-program-info/program/?program=${program}`,
        visitStatusUrl: `https://www.stsci.edu/jwst-program-info/visits/?program=${program}`,
        results
      });
    } catch (err: any) {
      console.error(`[API] Scraping Error for program ${program}:`, err);
      return res.status(500).json({ success: false, error: err.message });
    }
  });

  // Helper functions for parsing
  function convertDecimalToDate(year: number, dayOfYear: number): string {
    const date = new Date(year, 0, 1);
    date.setDate(date.getDate() + (dayOfYear - 1));
    const m = date.getMonth() + 1;
    const d = date.getDate();
    const yStr = String(date.getFullYear() % 100);
    return `${m}/${d}/${yStr}`;
  }

  function formatDateStr(date: Date): string {
    if (isNaN(date.getTime())) return "";
    const m = date.getMonth() + 1;
    const d = date.getDate();
    const yStr = String(date.getFullYear() % 100);
    return `${m}/${d}/${yStr}`;
  }

  // Vite middleware setup
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Server] running on http://localhost:${PORT}`);
  });
}

startServer();
