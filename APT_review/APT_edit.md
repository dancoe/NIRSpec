# 🧪 APT File Editing? (Experimental)

> ⚠️
> Editing JWST APT (.aptx) files using external scripts is inherently risky. The APT file format is a complex ZIP archive containing structured XML and metadata that the Astronomer's Proposal Tool (APT) maintains with strict formatting requirements. 
> 
> `APT_edit.py` is **experimental** and untested. So far it only edits a dither table for NIRSpec MOS.
>
> Always keep a backup of your original `.aptx` file before attempting any programmatic modifications.

## 🧠 Challenges and Solutions

### 1. The `.aptx` File Structure
An `.aptx` file is actually a ZIP archive. It typically contains:
- `manifest`: A small file specifying which XML file is the primary proposal.
- `{ProposalID}.xml`: The main body of the proposal.
- `edu.stsci.mpt...`: UI state files for the MSA Planning Tool (MPT).

### 2. XML Parser Sensitivity
Initial attempts used standard XML libraries like `xml.etree.ElementTree`. While these are excellent for most XML tasks, they were rejected by APT for several reasons:
- **Namespace Handling**: Libraries like `ElementTree` often rename or consolidate namespaces (e.g., adding `ns0:` prefixes), which can break APT's rigid internal parser.
- **Header Formatting**: APT expects specific XML headers, including `standalone="yes"` and often includes critical diagnostic comments at the top. Most XML libraries strip these out or alter them.
- **Sorting**: Standard parsers may alphabetically sort attributes or rearrange elements, which can cause validation failures in APT.

**Solution**: Use **Targeted Regex Pattern Matching** for specific values (like dither offsets) instead of full XML deserialization. This preserves 100% of the original file's formatting, headers, and comments.

### 3. ZIP Entry Order
The `manifest` file is usually the first entry in an original `.aptx` archive. Some Java-based ZIP implementations (like the one used in APT) benefit from this specific ordering to correctly identify the file type before reading the rest of the archive.

### 4. Backreference Ambiguity in Regex
When substituting numerical values (like `-0.185`) into an XML string using regex, backreferences like `\1` can become ambiguous if the substituted value starts with a digit (e.g., `\10.185` looks like a reference to capture group 10).

**Solution**: Always use the explicit group notation `\g<1>` for backreferences in Python's `re.sub` to avoid collisions with the replacement data.

---

## 🛠 Usage Recommendations
If you must modify an APT file programmatically:
1. **Target only what you need**: Avoid rebuilding the entire XML tree.
2. **Preserve order**: When rebuilding the ZIP, maintain the original file order.
3. **Validate early**: After modification, try opening the file in APT to check if it can still parse the file.

## 🚀 How to run
Run the script from the root of the repository:

```bash
python3 APT_edit.py <Program>.aptx --dithers <DitherFile>.txt
```

Example for Program 9278:
```bash
python3 APT_edit.py data/9278/JWST9278.aptx --dithers data/9278/JWST9278_dithers_zigzag.txt
```

By default, this will produce `data/9278/JWST9278_mod.aptx`. You can specify a different output filename using the `--output` flag:

```bash
python3 APT_edit.py JWST9278.aptx --dithers dithers.txt --output custom_name.aptx
```
