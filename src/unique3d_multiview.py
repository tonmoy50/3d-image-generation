import subprocess
import os


def main():
    os.chdir("./Unique3D")

    # Run the script as a module
    result = subprocess.run(
        ["python3", "-m", "app.custom_models.mvimg_prediction"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    main()
