#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#ifndef PORTABLE_PLOT_SCRIPT
#define PORTABLE_PLOT_SCRIPT "scripts/plot.py"
#endif

inline std::string portable_shell_quote(const std::string &s)
{
    std::string out = "'";
    for (char c : s)
    {
        if (c == '\'')
        {
            out += "'\\''";
        }
        else
        {
            out += c;
        }
    }
    out += "'";
    return out;
}

inline int run_plot_command(const std::string &launcher, const std::string &script_path, const std::string &csv_path)
{
    const std::string command =
        launcher + " " +
        portable_shell_quote(script_path) + " " +
        portable_shell_quote(csv_path);

    std::cout << "Plot command: " << command << '\n';
    return std::system(command.c_str());
}

inline bool plot_csv_with_portable_script(const std::string &csv_path)
{
    const std::string script_path = PORTABLE_PLOT_SCRIPT;

    int rc = run_plot_command("python3", script_path, csv_path);
    if (rc == 0)
    {
        return true;
    }

    rc = run_plot_command("python", script_path, csv_path);
    if (rc == 0)
    {
        return true;
    }

    std::cerr
        << "Portable could not launch its local plot script. "
        << "The CSV is still available at: " << csv_path << '\n';
    return false;
}
