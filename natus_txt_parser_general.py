import os
import re
import json
import numpy as np
import pandas as pd
from scipy.io import savemat


SECTION_RE = re.compile(r'^\[(.+?)\]\s*$')
KEYVAL_RE = re.compile(r'^([^=\n\r]+?)=(.*)$')


def try_parse_number(value):
    """
    Convierte números con coma decimal:
      '24,416667' -> 24.416667
      '100' -> 100
    Si no es número, devuelve el texto original.
    """
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return ""

    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', text):
        return text

    candidate = text.replace(",", ".")

    try:
        num = float(candidate)
        if num.is_integer():
            return int(num)
        return num
    except Exception:
        return text


def parse_signal_like_text(raw_text):
    """
    Reconstruye una señal exportada como:
        0,12,0,03,-0,01,.../0,00,0,02,...
    donde:
    - la coma actúa como decimal
    - la barra '/' separa continuación de línea
    Devuelve np.array de float.

    Si no consigue interpretar el contenido como señal, devuelve array vacío.
    """
    if raw_text is None:
        return np.array([], dtype=float)

    txt = str(raw_text).replace("\n", "").replace("\r", "")
    txt = txt.replace("/", ",").strip(",")

    if txt == "":
        return np.array([], dtype=float)

    tokens = [t.strip() for t in txt.split(",") if t.strip() != ""]

    values = []
    i = 0
    while i < len(tokens) - 1:
        a = tokens[i]
        b = tokens[i + 1]

        if re.fullmatch(r'[-+]?\d+', a) and re.fullmatch(r'\d+', b):
            try:
                values.append(float(f"{a}.{b}"))
                i += 2
                continue
            except Exception:
                pass

        i += 1

    if len(values) < 5:
        return np.array([], dtype=float)

    return np.array(values, dtype=float)


def sanitize_name(name):
    name = str(name).strip()
    name = name.replace("°", "deg")
    name = name.replace("%", "pct")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-\(\)\[\]\.<>]", "_", name)
    return name
def read_text_with_fallback_encodings(filepath, encodings=None):
    """
    Intenta leer el archivo con varias codificaciones y devuelve:
    - lines: lista de líneas
    - used_encoding: codificación utilizada
    """
    if encodings is None:
        encodings = ["utf-16", "utf-8", "latin-1"]

    last_error = None

    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc, errors="strict") as f:
                lines = f.readlines()

            has_section = any(SECTION_RE.match(line.strip()) for line in lines if line.strip())
            has_keyval = any(KEYVAL_RE.match(line.strip()) for line in lines if line.strip())

            if has_section or has_keyval:
                return lines, enc

        except Exception as e:
            last_error = e

    raise UnicodeError(
        f"No se pudo leer correctamente el archivo con las codificaciones probadas: {encodings}. "
        f"Último error: {last_error}"
    )

def parse_natus_txt(filepath):
    """
    Parser general para archivos .txt exportados por Natus.
    Detecta:
    - secciones [ ... ]
    - pares clave=valor
    - señales largas partidas en varias líneas
    """
    lines, used_encoding = read_text_with_fallback_encodings(filepath)
       
    header_comments = []
    sections = []

    current_section = None
    current_data = {}
    collecting_multiline_value = False
    multiline_key = None

    def flush_section():
        nonlocal current_section, current_data
        if current_section is not None:
            sections.append({
                "section_name": current_section,
                "data": current_data.copy()
            })

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n").rstrip("\r")
        line = raw.strip()

        if line == "":
            i += 1
            continue

        if line.startswith(";"):
            header_comments.append(line)
            i += 1
            continue

        msec = SECTION_RE.match(line)
        if msec:
            flush_section()
            current_section = msec.group(1).strip()
            current_data = {}
            collecting_multiline_value = False
            multiline_key = None
            i += 1
            continue

        if collecting_multiline_value:
            if SECTION_RE.match(line) or KEYVAL_RE.match(line):
                collecting_multiline_value = False
                multiline_key = None
                continue

            current_data[multiline_key] += line
            i += 1
            continue

        mkv = KEYVAL_RE.match(line)
        if mkv:
            key = mkv.group(1).strip()
            value = mkv.group(2).strip()
            current_data[key] = value

            signal_guess = (
                value.endswith("/") or
                "Datos promediados" in key or
                ("Datos" in key and "," in value)
            )

            if signal_guess and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and (not SECTION_RE.match(next_line)) and (not KEYVAL_RE.match(next_line)):
                    collecting_multiline_value = True
                    multiline_key = key
                elif value.endswith("/"):
                    collecting_multiline_value = True
                    multiline_key = key

            i += 1
            continue

        if multiline_key is not None:
            current_data[multiline_key] += line

        i += 1

    flush_section()

    parsed = {
        "source_file": filepath,
        "used_encoding": used_encoding,
        "header_comments": header_comments,
        "sections": sections
    }

    return parsed


def enrich_parsed_data(parsed):
    """
    Añade:
    - versión tipada de los datos
    - detección general de señales
    - resumen de bloques con señal
    """
    enriched_sections = []
    signal_blocks = []

    for sec_idx, sec in enumerate(parsed["sections"]):
        sec_name = sec["section_name"]
        raw_data = sec["data"]

        typed_data = {}
        local_signal_entries = []

        for k, v in raw_data.items():
            signal = parse_signal_like_text(v)

            if signal.size > 0:
                typed_data[k] = signal
                local_signal_entries.append({
                    "key": k,
                    "signal": signal,
                    "n_samples": int(signal.size)
                })
            else:
                typed_data[k] = try_parse_number(v)

        enriched_sections.append({
            "section_index": sec_idx + 1,
            "section_name": sec_name,
            "data": typed_data
        })

        if local_signal_entries:
            block_summary = {
                "section_index": sec_idx + 1,
                "section_name": sec_name,
                "signal_entries": []
            }

            channel = raw_data.get("Número del canal", None)
            sweeps = raw_data.get("Barridos", None)
            rejected = raw_data.get("Barridos rechazados", None)
            fs = raw_data.get("Frecuencia de muestreo(kHz)", None)
            duration = raw_data.get("Duración de barridos(ms)", None)

            block_summary["numero_canal"] = try_parse_number(channel) if channel is not None else None
            block_summary["barridos"] = try_parse_number(sweeps) if sweeps is not None else None
            block_summary["barridos_rechazados"] = try_parse_number(rejected) if rejected is not None else None
            block_summary["frecuencia_muestreo_khz"] = try_parse_number(fs) if fs is not None else None
            block_summary["duracion_barrido_ms"] = try_parse_number(duration) if duration is not None else None

            for entry in local_signal_entries:
                block_summary["signal_entries"].append({
                    "key": entry["key"],
                    "n_samples": entry["n_samples"],
                    "signal": entry["signal"]
                })

            signal_blocks.append(block_summary)

    enriched = {
        "source_file": parsed["source_file"],
        "header_comments": parsed["header_comments"],
        "sections": enriched_sections,
        "signal_blocks": signal_blocks
    }

    return enriched


def create_signal_summary_dataframe(enriched):
    rows = []

    for block in enriched["signal_blocks"]:
        for sig in block["signal_entries"]:
            rows.append({
                "section_index": block["section_index"],
                "section_name": block["section_name"],
                "signal_key": sig["key"],
                "n_samples": sig["n_samples"],
                "numero_canal": block.get("numero_canal"),
                "barridos": block.get("barridos"),
                "barridos_rechazados": block.get("barridos_rechazados"),
                "frecuencia_muestreo_khz": block.get("frecuencia_muestreo_khz"),
                "duracion_barrido_ms": block.get("duracion_barrido_ms"),
            })

    return pd.DataFrame(rows)


def export_parsed_outputs(enriched, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(enriched["source_file"]))[0]

    json_path = os.path.join(output_dir, f"{base_name}_parsed.json")

    json_ready = {
        "source_file": enriched["source_file"],
        "header_comments": enriched["header_comments"],
        "sections": [],
        "signal_blocks": []
    }

    for sec in enriched["sections"]:
        sec_data = {}
        for k, v in sec["data"].items():
            if isinstance(v, np.ndarray):
                sec_data[k] = v.tolist()
            else:
                sec_data[k] = v
        json_ready["sections"].append({
            "section_index": sec["section_index"],
            "section_name": sec["section_name"],
            "data": sec_data
        })

    for block in enriched["signal_blocks"]:
        b = dict(block)
        new_entries = []
        for sig in block["signal_entries"]:
            new_entries.append({
                "key": sig["key"],
                "n_samples": sig["n_samples"],
                "signal": sig["signal"].tolist()
            })
        b["signal_entries"] = new_entries
        json_ready["signal_blocks"].append(b)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, ensure_ascii=False, indent=2)

    summary_df = create_signal_summary_dataframe(enriched)
    csv_path = os.path.join(output_dir, f"{base_name}_signal_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    mat_path = os.path.join(output_dir, f"{base_name}_parsed.mat")

    mdict = {}

    if not summary_df.empty:
        for col in summary_df.columns:
            try:
                mdict[f"summary__{sanitize_name(col)}"] = summary_df[col].to_numpy(dtype=float)
            except Exception:
                mdict[f"summary__{sanitize_name(col)}"] = summary_df[col].astype(str).to_numpy(dtype=object)

    for i, block in enumerate(enriched["signal_blocks"], start=1):
        for j, sig in enumerate(block["signal_entries"], start=1):
            name = f"signal_block_{i}_entry_{j}"
            mdict[name] = sig["signal"]

            fs = block.get("frecuencia_muestreo_khz", None)
            if isinstance(fs, (int, float)) and not pd.isna(fs):
                time_ms = np.arange(len(sig["signal"])) / (fs * 1000.0) * 1000.0
                mdict[f"{name}_time_ms"] = time_ms

    savemat(mat_path, mdict)

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "mat_path": mat_path
    }


def parse_and_export_natus_txt(filepath, output_dir):
    parsed = parse_natus_txt(filepath)
    enriched = enrich_parsed_data(parsed)
    outputs = export_parsed_outputs(enriched, output_dir)
    return enriched, outputs


if __name__ == "__main__":
    input_txt = r"C:\RUTA\archivo_natus.txt"
    output_dir = r"C:\RUTA\salida_parser"

    enriched, outputs = parse_and_export_natus_txt(input_txt, output_dir)

    print(f"Archivo procesado: {input_txt}")
    print(f"Bloques detectados: {len(enriched['sections'])}")
    print(f"Bloques con señal: {len(enriched['signal_blocks'])}")
    print(outputs)
