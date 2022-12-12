import os
import openai
from pyparsing import OneOrMore, nestedExpr, ParseResults

QUANTIFIERS = ["forall", "exists"]

def to_unique_list(parsed):
    if(isinstance(parsed, ParseResults)):
        nlist = []
        if(parsed[0] in QUANTIFIERS):
            for o1 in parsed[1]:
                for o2 in parsed[1]:
                    if(o1 != o2):
                        nlist.append(["not", ["=", o1, o2]])
            return [to_unique_list(parsed[0]), 
                    to_unique_list(parsed[1]), 
                    ["and"] + to_unique_list(parsed[2:]) + nlist]
        else:
            for q in parsed:
                nlist.append(to_unique_list(q))
        return nlist
    elif(isinstance(parsed, list)):
        return [to_unique_list(q) for q in parsed]
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
        print("Translation: " + response_text)
        return to_unique_list(data)[0], response_text
    except:
        print("Error: invalid LISP")
        exit()




if __name__ == "__main__":
    text_to_goal("pick up a block")