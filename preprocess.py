from tika import parser  # Parsing pdf files
import re  # Regex
import os  # For listing all files in the directory
import pickle  # Loading and saving objects with pickle

# ------------------------------------------
#  PICKLE FUNCTIONS FOR SAVING AND LOADING


def save_obj(obj, name):
    """

    :param obj: object to be saved as a .pkl file
    :param name: name of the .pkl file saved
    """
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """

    :param name: name of the .pkl file to be loaded
    :return: variable containing the object loaded from the .pkl file
    """
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# ------------------------------------------
#  FUNCTION FOR REMOVING ACTIONS MIXED IN THE SAME LINE AS THE DIALOGUE


def clean_dialogue(dialogue):
    """

    :param dialogue: Raw text of the dialogue
    :return: Dialogue without any actions
    """
    dialogue = re.sub("[[][^]]+?[]]", '', dialogue)  # Removing strings enclosed in square brackets (e.g. [...])
    dialogue = re.sub("^[[][^]a-z]+", '', dialogue)  # Removing strings with starting square bracket (e.g. [...)
    dialogue = re.sub("[^]a-z]+[]]$", '', dialogue)  # Removing strings with ending square bracket (e.g. ...])

    return remove_newline(dialogue)

# ------------------------------------------
#  FUNCTION FOR REMOVING NEW LINE CHARACTERS IN THE DIALOGUE


def remove_newline(dialogue):
    """

    :param dialogue: Raw text of the dialogue
    :return: Dialogue without any new line characters
    """
    return re.sub('\s+', ' ', dialogue).strip()

# ------------------------------------------
#  FUNCTION FOR EXTRACTING ACTIONS IN THE DIALOGUE


def get_action(dialogue):
    """

    :param dialogue: Raw text of the dialogue
    :return: A list of all actions mixed in the dialogue
    """
    actions = []  # list of actions to be returned
    dialogue = remove_newline(dialogue)

    # Extracting strings enclosed in square brackets (e.g. [...])
    actions += re.findall("[[][^]]+?[]]", dialogue)
    dialogue = re.sub("[[][^]]+?[]]", '', dialogue)
    # Extracting strings with starting square bracket (e.g. [...)
    actions += re.findall("^[[][^]a-z]+", dialogue)
    dialogue = re.sub("^[[][^]a-z]+", '', dialogue)
    # Extracting strings with ending square bracket (e.g. ...])
    actions += re.findall("[^]a-z]+[]]$", dialogue)
    dialogue = re.sub("[^]a-z]+[]]$", '', dialogue)

    # Remove square brackets for all action strings
    actions = [re.sub('\[|\]', '', action) for action in actions]

    return actions

# ------------------------------------------
#  FUNCTION FOR UPDATING THE CHARACTERS AND ACTIONS LIST
#  SO THAT ACTIONS CAN BE RETROSPECTIVELY ASSIGNED TO THE PREVIOUS/NEXT CHARACTER'S DIALOGUE


def update_speaker(speaker, current_char, last1_char, last2_char, temp_actions, current_actions, last1_actions, last2_actions, push_action, combined_dictionary):
    """

    :param speaker: character being processed
    :param current_char: Current character
    :param last1_char: Last character
    :param last2_char: Second Last character
    :param temp_actions: Actions being processed
    :param current_actions: Current actions
    :param last1_actions: Last actions
    :param last2_actions: Second Last actions
    :param push_action: Flag to update actions
    :param combined_dictionary: Dictionary containing list of tuples of (dialogue, actions) for each character
    :return: updated character and actions list, push_action flag and the combined_dictionary
    """
    if push_action:
        # Update the actions list
        last2_actions[:] = last1_actions
        last1_actions[:] = current_actions
        current_actions[:] = temp_actions
        temp_actions.clear()
    if not push_action:
        # First character has finished their utterance, update the actions list on the next call to this function
        push_action = True

    # Retrospectively assign the last actions to the current character or the second last character
    matching_actions = []
    if last1_char != '':
        # Check whether the last action mentions the current character
        # if so append the actions to the current character's last dialogue
        if current_char != '' and last1_char != current_char:
            for action in last1_actions:
                if current_char in action:
                    matching_actions.append(action)
            combined_dictionary[current_char][-1] = (combined_dictionary[current_char][-1][0], combined_dictionary[current_char][-1][1] + matching_actions)

        # Check whether the last action mentions the second last character
        # if so append the actions to the second last character's last dialogue
        matching_actions.clear()
        if last2_char != '' and last1_char != last2_char and last2_char != current_char:
            for action in last1_actions:
                if last2_char in action:
                    matching_actions.append(action)
            combined_dictionary[last2_char][-1] = (combined_dictionary[last2_char][-1][0], combined_dictionary[last2_char][-1][1] + matching_actions)

    # Update the characters list
    last2_char = last1_char
    last1_char = current_char
    current_char = speaker

    return current_char, last1_char, last2_char, temp_actions, current_actions, last1_actions, last2_actions, push_action, combined_dictionary

# ------------------------------------------
#  FUNCTION TO PROCESS THE SCRIPT FILES AND LOAD A DICTIONARY CONTAINING LIST OF TUPLES (DIALOGUE, ACTIONS) FOR EACH CHARACTER


def load_dictionary(character_dictionary, combined_dictionary, path):
    """

    :param character_dictionary: Dictionary containing dialogues only
    :param combined_dictionary: Dictionary containing dialogues and actions
    :param path: Path to the script .pdf files
    :return: N/A
    """
    # Process all script files in the directory at path
    for filename in os.listdir(path):
        # Read the .pdf file using the tika parser
        filename = path + filename
        print(filename)
        raw = parser.from_file(filename)

        #  Three keys, status content and metadata in the dictionary read by the parser
        #  Content contains the main text of the script
        #  Split on double line end
        corpus = raw['content'].split("\n\n")

        # Extract utterances only
        filtered = []
        dialogue_started = False  # Keep track of when dialogue starts
        processing_header = False  # Keep track of when the headers (e.g. title, prog no) are being processed
        char_name_regex = re.compile('^[^a-z]+[:]') # To look for character names
        for line in corpus:
            # Skip until utterance starts (i.e. character name found)
            if not (dialogue_started) and char_name_regex.search(line):
                dialogue_started = True

            # Ignore headers on each page (Always at the end of the page, starts with tag 'Prog No' and ends wtih empty line)
            if not (processing_header) and line.startswith('Prog No'):
                processing_header = True
            elif processing_header and line == '':
                processing_header = False

            # Add utterance only to filtered list
            if dialogue_started and not (processing_header) and line != '':
                filtered.append(line)

        # Loop through the dialogue to populate the dictionary

        last2_char = ''  # Keep track of the character speaking before the last speaking character
        last1_char = ''  # Keep track of the last character speaking
        current_char = ''  # Keep track of the current character speaking

        push_action = False  # Flag to keep track of when the first character is done with their utterance
        last2_actions = []  # Keep track of actions of the speaker before the last speaker
        last1_actions = []  # Keep track of actions of last speaker
        current_actions = []  # Keep track of actions of current speaker
        temp_actions = []  # List of actions being processed

        #  For each element of the filtered list, split on :, save the first word as key, and append the utterance as value
        for dialogue in filtered:

            # Clear current characters and actions at the start of each scene
            if dialogue.startswith('SCENE'):
                current_char = ''
                last2_char = ''
                last1_char = ''
                current_actions.clear()
                last2_actions.clear()
                last1_actions.clear()
                temp_actions.clear()
                push_action = False

            # If line is in all caps, must be actions, no need to look for character name
            if re.match("^[^a-z]+$", dialogue):
                # Only proceed if there is a current speaker (i.e. do not process scene description)
                if current_char != '' and ':' not in dialogue:
                    # Update dictionaries
                    if current_char in combined_dictionary:
                        # Existing character
                        combined_dictionary[current_char][-1] = (combined_dictionary[current_char][-1][0], combined_dictionary[current_char][-1][1] + [re.sub('\[|\]', '', remove_newline(dialogue))])
                        temp_actions.append(re.sub('\[|\]', '', remove_newline(dialogue)))
                    else:
                        # New character
                        combined_dictionary[current_char] = [('', [re.sub('\[|\]', '', remove_newline(dialogue))])]
                        temp_actions.append(re.sub('\[|\]', '', remove_newline(dialogue)))
                continue

            # Remove ":" within square brackets
            dialogue = re.sub(r"(\[[^\]]*):([^\[]*\])", r"\1\2", dialogue)
            # Split on ":" to extract character names
            split_dialogue = dialogue.split(':')

            # Now we extract characters, dialogues and actions

            dialogue_final = ''
            action_final = []

            # Exactly one character and one utterance
            if len(split_dialogue) == 2:
                # Extract dialogue and actions
                dialogue_final = clean_dialogue(split_dialogue[1])
                action_final = get_action(split_dialogue[1])
                # Update characters and actions list, and retrospectively assign actions to previous/next character
                current_char, last1_char, last2_char, temp_actions, current_actions, last1_actions, last2_actions, push_action, combined_dictionary = update_speaker(split_dialogue[0], current_char, last1_char, last2_char, temp_actions, current_actions, last1_actions, last2_actions, push_action, combined_dictionary)
                # Keep track of actions of character being processed
                temp_actions += action_final

                # Update dictionaries
                if split_dialogue[0] in character_dictionary:
                    # Existing character
                    character_dictionary[current_char].append(dialogue_final)
                    combined_dictionary[current_char].append((dialogue_final, action_final))
                else:
                    # New character
                    character_dictionary[current_char] = [dialogue_final]
                    combined_dictionary[current_char] = [(dialogue_final, action_final)]

            # Continuing utterance, add to the last speaking character
            elif len(split_dialogue) == 1 and current_char != '':
                # Extract dialogue and actions
                dialogue_final = clean_dialogue(split_dialogue[0])
                action_final = get_action(split_dialogue[0])
                # Keep track of actions of character being processed
                temp_actions += action_final

                # Update dictionaries
                character_dictionary[current_char].append(dialogue_final)
                combined_dictionary[current_char].append((dialogue_final, action_final))
            # Dialogue with multiple charaters speaking or colons used in the utterance (e.g. bible quotes, lists, etc.)
            elif len(split_dialogue) > 2:
                for i in split_dialogue:

                    # Text contains charater name only
                    if re.match('^[A-Z]+$', i):
                        # Update characters and actions list, and retrospectively assign actions to previous/next character
                        current_char, last1_char, last2_char, temp_actions, current_actions, last1_actions, last2_actions, push_action, combined_dictionary = update_speaker(i, current_char, last1_char, last2_char, temp_actions, current_actions, last1_actions, last2_actions, push_action, combined_dictionary)

                    # Text contains utterance and next character
                    elif i != '' and i[-1].isupper():
                        # Extract dialogue and actions
                        dialogue_final = clean_dialogue(i.rsplit('\n', 1)[0])
                        action_final = get_action(i.rsplit('\n', 1)[0])

                        # Update dictionaries
                        if current_char in character_dictionary:
                            # Existing character
                            character_dictionary[current_char].append(dialogue_final)
                            combined_dictionary[current_char].append((dialogue_final, action_final))
                        else:
                            # New character
                            character_dictionary[current_char] = [dialogue_final]
                            combined_dictionary[current_char] = [(dialogue_final, action_final)]

                        # Update characters and actions list, and retrospectively assign actions to previous/next character
                        current_char, last1_char, last2_char, temp_actions, current_actions, last1_actions, last2_actions, push_action, combined_dictionary = update_speaker(i.rsplit('\n', 1)[1], current_char, last1_char, last2_char, temp_actions, current_actions, last1_actions, last2_actions, push_action, combined_dictionary)

                        # Keep track of actions of character being processed
                        temp_actions += action_final

                    # Text contains utterance only, add to last character
                    else:
                        # Extract dialogue and actions
                        dialogue_final = clean_dialogue(i)
                        action_final = get_action(i)
                        # Keep track of actions of character being processed
                        temp_actions += action_final

                        # Update dictionaries
                        if current_char in character_dictionary:
                            # Existing character
                            character_dictionary[current_char].append(dialogue_final)
                            combined_dictionary[current_char].append((dialogue_final, action_final))
                        else:
                            # New character
                            character_dictionary[current_char] = [dialogue_final]
                            combined_dictionary[current_char] = [(dialogue_final, action_final)]

        #  From the dictionaries, pop any key that is anything other than a NAME
        regex = re.compile(r'[A-Z]{2,}')
        for key in character_dictionary.copy():
            # After processing, it has been observed that some keys with ] and / are going through. Removing these.
            if not regex.match(key) or ' ' in key or ']' in key or '/' in key:
                character_dictionary.pop(key)

        for key in combined_dictionary.copy():
            # After processing, it has been observed that some keys with ] and / are going through. Removing these.
            if not regex.match(key) or ' ' in key or ']' in key or '/' in key:
                combined_dictionary.pop(key)

# ------------------------------------------
#  FUNCTION TO REMOVE CHARACTERS THAT HAS LESS THAN N UTTERANCES


def remove_minor_characters(character_dictionary, N):
    """

    :param character_dictionary: Dictionary containing utterance data for each character
    :param N: Threshold on the number of utternaces
    :return: N/A
    """
    # If N!= -1 , from the dictionary, pop any key that has less than N utterances
    if N != -1:
        for key in character_dictionary.copy():
            if len(character_dictionary[key]) < N:
                character_dictionary.pop(key)

# ------------------------------------------
#  FUNCTION TO REMOVE CHARACTERS THAT APPEARS IN THE TEST DATA BUT NOT THE TRAIN DATA


def synchronize_dictionaries(train_dictionary, test_dictionary):
    """

    :param train_dictionary: Dictionary containing train utterance data for each character
    :param test_dictionary: Dictionary containing test utterance data for each character
    :return:
    """
    for key in test_dictionary.copy():
        if key not in train_dictionary.keys():
            test_dictionary.pop(key)


# ------------------------------------------
#  Global Declarations

#  Path for all files
train_path = 'EastEnders_2008_1350-1399/train/'
test_path = 'EastEnders_2008_1350-1399/test/'

#  Global dictionary for characters and utterances
train_dictionary = {}
train_combined = {}
test_dictionary = {}
test_combined = {}
# ------------------------------------------
#  Main
load_dictionary(train_dictionary, train_combined, train_path)
remove_minor_characters(train_dictionary, 20)
remove_minor_characters(train_combined, 20)
save_obj(train_dictionary, "train_dictionary")
save_obj(train_combined, "train_combined")
load_dictionary(test_dictionary, test_combined, test_path)
remove_minor_characters(test_dictionary, 10)
remove_minor_characters(test_combined, 10)
synchronize_dictionaries(train_dictionary, test_dictionary)
save_obj(test_dictionary, "test_dictionary")
save_obj(test_combined, "test_combined")

# All character names
print('Train Characters: ')
print(sorted(train_combined.keys()))
print('Test Characters: ')
print(sorted(test_combined.keys()))
