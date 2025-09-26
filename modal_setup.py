import subprocess
import sys

def run_command(command):
    """Runs a command and prints its output."""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, executable="/bin/bash")
        for line in iter(process.stdout.readline, b''):
            print(line.decode('utf-8'), end='')
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except Exception as e:
        print(f"Error running command: {command}\n{e}")
        sys.exit(1)

def main():
    """Installs required python packages."""
    print("Installing required Python packages...")
    run_command("pip install modal pyyaml")
    print("Dependencies installed.")

if __name__ == "__main__":
    main()