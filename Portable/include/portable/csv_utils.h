#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

inline bool ensure_parent_directory(const std::string &path_text)
{
    const std::filesystem::path path(path_text);
    const std::filesystem::path parent = path.parent_path();
    if (parent.empty())
    {
        return true;
    }

    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    return !ec;
}

inline bool save_arrays_to_csv(
    const std::string &path_text,
    const std::vector<std::string> &headers,
    const std::vector<const std::vector<float> *> &columns)
{
    if (headers.empty() || headers.size() != columns.size())
    {
        return false;
    }

    if (!ensure_parent_directory(path_text))
    {
        return false;
    }

    std::ofstream out(path_text);
    if (!out.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < headers.size(); ++i)
    {
        out << headers[i];
        if (i + 1 < headers.size())
        {
            out << ';';
        }
    }
    out << '\n';

    size_t max_rows = 0;
    for (const std::vector<float> *column : columns)
    {
        if (column)
        {
            max_rows = std::max(max_rows, column->size());
        }
    }

    for (size_t row = 0; row < max_rows; ++row)
    {
        for (size_t col = 0; col < columns.size(); ++col)
        {
            const std::vector<float> *column = columns[col];
            if (column && row < column->size())
            {
                out << (*column)[row];
            }
            if (col + 1 < columns.size())
            {
                out << ';';
            }
        }
        out << '\n';
    }

    return true;
}
