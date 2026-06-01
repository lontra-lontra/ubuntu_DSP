#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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

inline std::string build_plot_command(
    const std::string &launcher,
    const std::string &script_path,
    const std::vector<std::string> &script_args)
{
    std::string command =
        launcher + " " +
        portable_shell_quote(script_path);

    for (const std::string &arg : script_args)
    {
        command += " " + portable_shell_quote(arg);
    }

    return command;
}

inline int run_plot_command(
    const std::string &launcher,
    const std::string &script_path,
    const std::vector<std::string> &script_args,
    bool print_command = true)
{
    const std::string command =
        build_plot_command(launcher, script_path, script_args);

    if (print_command)
    {
        std::cout << "Plot command: " << command << '\n';
    }
    return std::system(command.c_str());
}

inline int run_plot_command(
    const std::string &launcher,
    const std::string &script_path,
    const std::string &csv_path)
{
    return run_plot_command(
        launcher,
        script_path,
        std::vector<std::string>{csv_path});
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
