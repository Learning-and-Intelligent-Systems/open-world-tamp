import os
import openai
from pyparsing import OneOrMore, nestedExpr, ParseResults

def to_list(parsed):
    if(isinstance(parsed, ParseResults)):
        nlist = []
        for q in parsed:
            nlist.append(to_list(q))
        return nlist
    else:
        return parsed

def text_to_goal(command):
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open('/Users/aidancurtis/open-world-tamp/open_world/nlp/prompt.txt') as f:
        prompt = f.read()


    combined = prompt+"Q: "+command+"\n"
    response = openai.Completion.create(model="text-davinci-003", prompt=combined, temperature=0, max_tokens=128)

    try:
        response_text = response['choices'][0]['text'].replace("A:", "").replace("\n", "")
        data = OneOrMore(nestedExpr()).parseString(response_text)
        return to_list(data)[0], response_text
    except:
        print("Error: invalid LISP")
        exit()




if __name__ == "__main__":
    text_to_goal("pick up a block")