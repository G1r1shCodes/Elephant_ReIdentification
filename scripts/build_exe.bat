
@echo off
echo ========================================================
echo 🐘 BUILDING ELEPHANT RE-ID EXECUTABLE
echo ========================================================
echo This process may take a few minutes.
echo It will bundle Python, PyTorch, and Streamlit into a standalone folder.

REM Install PyInstaller if not present
pip install pyinstaller shimmy

REM Clean previous builds
rmdir /s /q build
rmdir /s /q dist

REM Run PyInstaller
REM --onedir: Create a folder (faster startup than --onefile)
REM --name: ElephantID_System
REM --hidden-import: Streamlit needs many hidden imports
REM --add-data: Include app.py and config files
REM --noconfirm: Don't ask to overwrite

pyinstaller --noconfirm --onedir --name "ElephantID_System" ^
    --hidden-import=streamlit ^
    --hidden-import=streamlit.web.cli ^
    --hidden-import=pandas ^
    --hidden-import=numpy ^
    --hidden-import=PIL ^
    --hidden-import=torch ^
    --hidden-import=torchvision ^
    --hidden-import=sklearn.utils._typedefs ^
    --hidden-import=sklearn.neighbors._partition_nodes ^
    --collect-all streamlit ^
    --copy-metadata streamlit ^
    --copy-metadata torch ^
    --copy-metadata tqdm ^
    --copy-metadata regex ^
    --copy-metadata requests ^
    --copy-metadata packaging ^
    --copy-metadata filelock ^
    --copy-metadata numpy ^
    --add-data "app.py;." ^
    --add-data "src;src" ^
    --add-data ".streamlit;.streamlit" ^
    --add-data "makhna_model.pth;." ^
    --add-data "gallery_embeddings.pt;." ^
    scripts\run_app_wrapper.py

echo.
echo ========================================================
echo ✅ BUILD COMPLETE
echo ========================================================
echo.
echo To run the app:
echo 1. Run 'dist\ElephantID_System\ElephantID_System.exe'
echo.
echo Note: Model and gallery files are now bundled automatically!
echo.
pause
