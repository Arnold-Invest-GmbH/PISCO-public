/* Keep bundled Poppler libraries local to Poppler, not the Python process. */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    char path[PATH_MAX], binary[PATH_MAX], libraries[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path)-1);
    if (n < 0 || n >= sizeof(path)-1) return 1; path[n] = 0;
    char *end = strrchr(path, '/'); if (!end) return 1; *end = 0;
    if (snprintf(binary, sizeof(binary), "%s/pdftoppm.real", path) >= sizeof(binary) ||
        snprintf(libraries, sizeof(libraries), "%s/../lib", path) >= sizeof(libraries)) return 1;
    setenv("LD_LIBRARY_PATH", libraries, 1); unsetenv("LD_PRELOAD");
    argv[0] = binary; execv(binary, argv);
    perror("Could not start bundled PDF renderer"); return 1;
}
