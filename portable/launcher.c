/* Native launcher: no shell, no installation, no system Python lookup.
 * The same unchanged Python sources run in the included isolated interpreter.
 */
#ifdef _WIN32
#define UNICODE
#include <windows.h>
#include <shellapi.h>
#include <wchar.h>
#include <stdlib.h>
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command, int show) {
    wchar_t root[32768], python[32768], script[32768];
    DWORD size = GetModuleFileNameW(NULL, root, 32768);
    if (!size || size >= 32768) return 1;
    wchar_t *end = wcsrchr(root, L'\\'); if (!end) return 1; *end = 0;
    swprintf(python, 32768, L"%ls\\runtime\\python.exe", root);
    swprintf(script, 32768, L"%ls\\app\\pisco_app.py", root);
    SetEnvironmentVariableW(L"PYTHONHOME", NULL); SetEnvironmentVariableW(L"PYTHONPATH", NULL);
    SetEnvironmentVariableW(L"TCL_LIBRARY", NULL); SetEnvironmentVariableW(L"TK_LIBRARY", NULL);
    size_t length = wcslen(python) + wcslen(script) + wcslen(command) + 32;
    wchar_t *line = calloc(length, sizeof(wchar_t)); if (!line) return 1;
    swprintf(line, length, L"\"%ls\" -I -B \"%ls\" %ls", python, script, command);
    STARTUPINFOW startup = {0}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION process = {0};
    BOOL ok = CreateProcessW(python, line, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, root, &startup, &process);
    free(line);
    if (!ok) {
        MessageBoxW(NULL, L"The supplied Python runtime could not be started. Extract the complete package first.", L"PISCO", MB_ICONERROR);
        return 1;
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1; GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess); CloseHandle(process.hThread); return (int)code;
}
#else
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
int main(int argc, char **argv) {
    char path[PATH_MAX], python[PATH_MAX], script[PATH_MAX], libraries[PATH_MAX];
#ifdef __APPLE__
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size)) return 1;
#else
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path)-1);
    if (n < 0 || n >= sizeof(path)-1) return 1;
    path[n] = 0;
#endif
    char *end = strrchr(path, '/'); if (!end) return 1; *end = 0;
    if (snprintf(python, sizeof(python), "%s/runtime/bin/python3", path) >= sizeof(python) ||
        snprintf(script, sizeof(script), "%s/app/pisco_app.py", path) >= sizeof(script)) return 1;
    const char *names[] = {"PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "TCL_LIBRARY", "TK_LIBRARY", NULL};
    for (int i=0; names[i]; i++) unsetenv(names[i]);
    if (snprintf(libraries, sizeof(libraries), "%s/tools/lib", path) >= sizeof(libraries)) return 1;
    setenv("LD_LIBRARY_PATH", libraries, 1);
    char **arguments = calloc(argc + 5, sizeof(char*)); if (!arguments) return 1;
    arguments[0] = python; arguments[1] = "-I"; arguments[2] = "-B"; arguments[3] = script;
    for (int i=1; i<argc; i++) arguments[i+3] = argv[i];
    execv(python, arguments);
    fprintf(stderr, "PISCO could not start its supplied runtime: %s. Extract the complete package.\n", strerror(errno));
    free(arguments); return 1;
}
#endif
