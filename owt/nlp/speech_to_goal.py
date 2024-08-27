import speech_recognition as sr
from open_world.nlp.text_to_goal import text_to_goal


def get_goal_audio():
    while True:
        # get audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("What would you like me to do?")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said: " + text)
            robot_goal, direct = text_to_goal(text)
            return robot_goal

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))


if __name__ == "__main__":
    get_goal_audio()
