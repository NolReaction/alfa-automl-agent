/* Alfa Lab Windows x64 launcher. No CRT, shell, network, or external downloads. */
typedef unsigned short WCHAR;
typedef unsigned int DWORD;
typedef unsigned short WORD;
typedef void *HANDLE;
typedef int BOOL;
#define API __attribute__((ms_abi))
#define MAX_PATH_CHARS 32768
#define INVALID_ATTRIBUTES 0xffffffffU

typedef struct {
    DWORD cb;
    WCHAR *lpReserved, *lpDesktop, *lpTitle;
    DWORD dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars;
    DWORD dwFillAttribute, dwFlags;
    WORD wShowWindow, cbReserved2;
    unsigned char *lpReserved2;
    HANDLE hStdInput, hStdOutput, hStdError;
} STARTUPINFOW;
typedef struct {
    HANDLE hProcess, hThread;
    DWORD dwProcessId, dwThreadId;
} PROCESS_INFORMATION;

extern DWORD API GetModuleFileNameW(HANDLE, WCHAR *, DWORD);
extern DWORD API GetFileAttributesW(const WCHAR *);
extern BOOL API CreateProcessW(const WCHAR *, WCHAR *, void *, void *, BOOL, DWORD,
                              void *, const WCHAR *, STARTUPINFOW *, PROCESS_INFORMATION *);
extern BOOL API CloseHandle(HANDLE);
extern void API ExitProcess(DWORD);
extern int API MessageBoxW(HANDLE, const WCHAR *, const WCHAR *, unsigned int);

static WCHAR app_dir[MAX_PATH_CHARS];
static WCHAR project_dir[MAX_PATH_CHARS];
static WCHAR python[MAX_PATH_CHARS];
static WCHAR desktop[MAX_PATH_CHARS];
static WCHAR command[MAX_PATH_CHARS];
static STARTUPINFOW startup;
static PROCESS_INFORMATION process;

static unsigned int length(const WCHAR *s) {
    unsigned int n = 0;
    while (s[n]) n++;
    return n;
}
static BOOL append(WCHAR *destination, const WCHAR *suffix) {
    unsigned int n = length(destination), m = length(suffix), i;
    if (n + m >= MAX_PATH_CHARS) return 0;
    for (i = 0; i <= m; i++) destination[n + i] = suffix[i];
    return 1;
}
static BOOL parent(WCHAR *path) {
    unsigned int i = length(path);
    while (i > 0) {
        i--;
        if (path[i] == '\\' || path[i] == '/') {
            path[i] = 0;
            return 1;
        }
    }
    return 0;
}
static BOOL file_exists(const WCHAR *path) {
    DWORD attributes = GetFileAttributesW(path);
    return attributes != INVALID_ATTRIBUTES && !(attributes & 0x10);
}
static void fail(const WCHAR *message) {
    MessageBoxW(0, message, (const WCHAR *)L"Alfa Lab", 0x10);
    ExitProcess(1);
}

void API start(void) {
    DWORD count = GetModuleFileNameW(0, app_dir, MAX_PATH_CHARS);
    if (!count || count >= MAX_PATH_CHARS || !parent(app_dir))
        fail((const WCHAR *)L"Не удалось определить папку приложения.");
    if (!append(project_dir, app_dir) || !parent(project_dir) ||
        !append(desktop, app_dir) || !append(desktop, (const WCHAR *)L"\\desktop.py"))
        fail((const WCHAR *)L"Слишком длинный путь к проекту. Переместите проект в более короткий путь.");
    if (!file_exists(desktop))
        fail((const WCHAR *)L"Файл app\\desktop.py не найден. Скопируйте папку app целиком в проект и запускайте AlfaLab.exe внутри неё.");
    if (!append(python, project_dir) ||
        !append(python, (const WCHAR *)L"\\.venv\\Scripts\\pythonw.exe"))
        fail((const WCHAR *)L"Слишком длинный путь к Python.");
    if (!file_exists(python)) {
        python[0] = 0;
        if (!append(python, project_dir) ||
            !append(python, (const WCHAR *)L"\\.venv\\Scripts\\python.exe") ||
            !file_exists(python))
            fail((const WCHAR *)L"В проекте не найдено окружение .venv с Python.\n\nЭтот EXE запускает установленную лабораторию. Сначала подготовьте Python 3.12 и зависимости проекта по инструкции. Затем повторите запуск.");
    }
    if (!append(command, (const WCHAR *)L"\"") || !append(command, python) ||
        !append(command, (const WCHAR *)L"\" -I \"") || !append(command, desktop) ||
        !append(command, (const WCHAR *)L"\""))
        fail((const WCHAR *)L"Слишком длинный путь к проекту.");
    startup.cb = sizeof(startup);
    if (!CreateProcessW(python, command, 0, 0, 0, 0x08000000, 0, project_dir, &startup, &process))
        fail((const WCHAR *)L"Не удалось запустить Python из .venv. Проверьте окружение проекта и права доступа к файлам.");
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    ExitProcess(0);
}
