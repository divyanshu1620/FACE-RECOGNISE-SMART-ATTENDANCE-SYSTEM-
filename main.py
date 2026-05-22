import os
import check_camera
import capture_image
import train_image
import recognize


def title_bar():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*50)
    print("  Smart Face Recognition Attendance System")
    print("="*50 + "\n")


def mainMenu():
    while True:
        title_bar()

        print("  [1]  Check Camera")
        print("  [2]  Capture Faces (Register Student)")
        print("  [3]  Train Model")
        print("  [4]  Recognize & Mark Attendance")
        print("  [5]  Exit")
        print()

        try:
            choice = int(input("  Enter Choice: "))
            print()

            if choice == 1:
                check_camera.camer()

            elif choice == 2:
                print("Face Capture")
                print("   • Follow the on-screen stage instructions")
                print("   • Each stage = different angle/lighting")
                print("   • Stay in good light, face clearly visible\n")
                capture_image.takeImages()

            elif choice == 3:
                print(" Training Model")
                print("   • This may take 30–60 seconds depending on image count\n")
                train_image.trainImages()

            elif choice == 4:
                print("🎥 Starting Attendance Recognition")
                print("   • Stand in front of camera")
                print("   • Wait for your name to appear in GREEN")
                print("   • Press Q when done\n")
                recognize.recognize_attendence()

            elif choice == 5:
                print("EXITED")
                break

            else:
                print("   Invalid choice. Enter 1–5.")

            input("\n  Press Enter to return to menu...")

        except ValueError:
            print("  ⚠ Please enter a valid number.")
            input("\n  Press Enter to continue...")
        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye.")
            break


if __name__ == "__main__":
    mainMenu()