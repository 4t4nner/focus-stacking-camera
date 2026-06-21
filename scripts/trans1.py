import json
import sys


def load_translations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)  # словарь: "номер_строки" -> новый_текст_строки


def main():
    if len(sys.argv) != 3:
        print("Usage: python trans1.py <translations.json> <target_file>")
        sys.exit(1)

    json_path = sys.argv[1]
    target_path = sys.argv[2]

    replacements = load_translations(json_path)

    with open(target_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for key, new_text in replacements.items():
        line_no = int(key)
        idx = line_no - 1  # нумерация строк с 1
        if idx < 0 or idx >= len(lines):
            print(f"WARNING: line {line_no} out of range (file has {len(lines)} lines)")
            continue
        # Сохраняем оригинальный перевод строки (\n)
        eol = "\n" if lines[idx].endswith("\n") else ""
        lines[idx] = new_text + eol
        print(f"Replaced line {line_no}")

    with open(target_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("Done.")


if __name__ == "__main__":
    main()