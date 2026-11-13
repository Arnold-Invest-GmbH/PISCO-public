/* Windows/macOS native entry points. All Python paths are bundle-relative. */
#ifdef _WIN32
#define UNICODE
#include <windows.h>
#include <wchar.h>
#include <stdlib.h>
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command, int show) {
    wchar_t root[32768], python[32768], script[32768];
    DWORD size = GetModuleFileNameW(NULL, root, 32768);
    if (!size || size >= 32768) return 1;
    wchar_t *end = wcsrchr(root, L'\\'); if (!end) return 1; *end = 0;
#ifdef PISCO_RASTERIZER
    for (int i=0; i<2; i++) { end=wcsrchr(root,L'\\'); if(!end) return 1; *end=0; }
    swprintf(script,32768,L"%ls\\app\\rasterize_pdf.py",root);
#else
    swprintf(script,32768,L"%ls\\app\\pisco_app.py",root);
#endif
    swprintf(python,32768,L"%ls\\runtime\\python.exe",root);
    SetEnvironmentVariableW(L"PYTHONHOME",NULL); SetEnvironmentVariableW(L"PYTHONPATH",NULL);
    size_t length=wcslen(python)+wcslen(script)+wcslen(command)+64;
    wchar_t *line=calloc(length,sizeof(wchar_t)); if(!line) return 1;
    swprintf(line,length,L"\"%ls\" -I -B -X utf8 \"%ls\" %ls",python,script,command);
    STARTUPINFOW startup={0}; startup.cb=sizeof(startup);
    startup.dwFlags=STARTF_USESTDHANDLES;
    startup.hStdInput=GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput=GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError=GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process={0};
    BOOL ok=CreateProcessW(python,line,NULL,NULL,TRUE,CREATE_NO_WINDOW,NULL,root,&startup,&process);
    free(line);
    if(!ok) {
        MessageBoxW(NULL,L"PISCO could not start its included runtime. Extract the complete archive first.",L"PISCO",MB_ICONERROR);
        return 1;
    }
    WaitForSingleObject(process.hProcess,INFINITE);
    DWORD code=1; GetExitCodeProcess(process.hProcess,&code);
    CloseHandle(process.hProcess); CloseHandle(process.hThread); return (int)code;
}
#else
#include <mach-o/dyld.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc,char **argv) {
    char path[PATH_MAX], root[PATH_MAX], python[PATH_MAX], script[PATH_MAX];
    uint32_t size=sizeof(path);
    if(_NSGetExecutablePath(path,&size)) return 1;
    char *end=strrchr(path,'/'); if(!end) return 1; *end=0;
#ifdef PISCO_RASTERIZER
    for(int i=0;i<2;i++) { end=strrchr(path,'/'); if(!end) return 1; *end=0; }
    if(!realpath(path,root)) return 1;
    const char *entry="rasterize_pdf.py";
#else
    char resources[PATH_MAX];
    if(snprintf(resources,sizeof(resources),"%s/../Resources",path)>=sizeof(resources)) return 1;
    if(!realpath(resources,root)) return 1;
    const char *entry="pisco_app.py";
#endif
    if(snprintf(python,sizeof(python),"%s/runtime/bin/python3",root)>=sizeof(python) ||
       snprintf(script,sizeof(script),"%s/app/%s",root,entry)>=sizeof(script)) return 1;
    const char *names[]={"PYTHONHOME","PYTHONPATH","DYLD_LIBRARY_PATH","DYLD_INSERT_LIBRARIES",NULL};
    for(int i=0;names[i];i++) unsetenv(names[i]);
    char **arguments=calloc(argc+7,sizeof(char*)); if(!arguments) return 1;
    arguments[0]=python; arguments[1]="-I"; arguments[2]="-B";
    arguments[3]="-X"; arguments[4]="utf8"; arguments[5]=script;
    for(int i=1;i<argc;i++) arguments[i+5]=argv[i];
    execv(python,arguments);
    fprintf(stderr,"PISCO could not start its supplied runtime: %s\n",strerror(errno));
    free(arguments); return 1;
}
#endif
