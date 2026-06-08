import os
import glob
import shutil

def brute_force_inject():
    # We know exactly where the destination is
    dest_dir = r"C:\Program Files\Python312\Lib\site-packages\llama_cpp\lib"
    
    # We know exactly where the source is based on your pip logs
    appdata = os.environ.get("APPDATA")
    source_dir = os.path.join(appdata, "Python", "Python312", "site-packages", "nvidia")

    print(f"Targeting Llama.cpp folder: {dest_dir}")
    print(f"Hunting in NVIDIA folder: {source_dir}")

    if not os.path.exists(source_dir):
        print("CRITICAL: Could not find the NVIDIA folder in your AppData.")
        return

    dlls = glob.glob(os.path.join(source_dir, "**", "*.dll"), recursive=True)
    
    copied = 0
    for dll_path in dlls:
        try:
            shutil.copy(dll_path, dest_dir)
            print(f" -> Successfully copied: {os.path.basename(dll_path)}")
            copied += 1
        except Exception as e:
            print(f"Failed to copy {os.path.basename(dll_path)}: {e}")

    if copied > 0:
        print(f"\nSUCCESS! {copied} DLLs brute-forced into the llama.cpp directory.")
    else:
        print("\nWARNING: Found the NVIDIA folder, but no DLLs were inside it.")

if __name__ == "__main__":
    brute_force_inject()