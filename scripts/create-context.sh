#!/bin/bash

# Определяем корень проекта относительно расположения скрипта
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

INPUT_FILE="$PROJECT_ROOT/input.txt"
CONTEXT_FILE="$PROJECT_ROOT/context.md"

# Проверяем наличие input.txt и создаём его при отсутствии
if [ ! -f "$INPUT_FILE" ]; then
    touch "$INPUT_FILE"
    echo "Файл создан: $INPUT_FILE"
    echo "Пожалуйста, заполните его относительными путями к файлам (пример: ./folder/file.txt)"
    exit 1
fi

# Очищаем или создаём контекстный файл
> "$CONTEXT_FILE"

# Функция добавления одного файла в контекст
add_file() {
    local file_path="$1"
    local rel_path
    rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$file_path")

    {
        echo "$rel_path:"
        echo ''
        echo '```'
        cat "$file_path"
        echo ''
        echo '```'
        echo ''
    } >> "$CONTEXT_FILE"

    echo "✅ Добавлен файл: $rel_path"
}

# Функция добавления всех файлов из папки (без рекурсии)
add_dir() {
    local dir_path="$1"
    local rel_dir
    rel_dir=$(realpath --relative-to="$PROJECT_ROOT" "$dir_path")
    echo "📁 Папка: $rel_dir — добавляю файлы..."

    local count=0
    for f in "$dir_path"/*; do
        [ -f "$f" ] || continue
        add_file "$f"
        ((count++))
    done

    if [ "$count" -eq 0 ]; then
        echo "⚠️ Папка пуста: $rel_dir" >&2
    fi
}

# Обрабатываем каждый путь из input.txt
while IFS= read -r line || [ -n "$line" ]; do
    line=$(echo "$line" | tr -d '\r')
    [ -z "$line" ] && continue

    rel_path=$(echo "$line" | sed 's|^\./||')
    full_path="$PROJECT_ROOT/$rel_path"

    # 1. Точное совпадение — файл
    if [ -f "$full_path" ]; then
        add_file "$full_path"
        continue
    fi

    # 2. Точное совпадение — папка
    if [ -d "$full_path" ]; then
        add_dir "$full_path"
        continue
    fi

    # 3. Glob: добавить один символ к расширению (например .ts → .tsx)
    match=$(ls -1 "${full_path}"? 2>/dev/null | head -1)
    if [ -n "$match" ] && [ -f "$match" ]; then
        echo "🔍 Найден с расширением: $(realpath --relative-to="$PROJECT_ROOT" "$match")"
        add_file "$match"
        continue
    fi

    # 4. Путь без расширения — проверяем, является ли папкой
    # Убираем расширение: store/ProjectStore.ts → store/ProjectStore
    no_ext="${full_path%.*}"
    if [ -d "$no_ext" ]; then
        add_dir "$no_ext"
        continue
    fi

    echo "⚠️ Файл не найден: $rel_path" >&2
done < "$INPUT_FILE"

echo -e "\nКонтекст сохранён в: ./context.md\n" 


