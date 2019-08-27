import pickle            # Loading and saving objects with pickle
import pandas as pd

# ------------------------------------------
#  PICKLE FUNCTIONS FOR SAVING AND LOADING OBJECT FROM THE FILES


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# ------------------------------------------
# CONVERTING CHARACTER DICTIONARIES INTO DATAFRAMES SUITABLE FOR THE PROCESSING
combined = True # If set to True - include the actions into the data frame. If set to False - do not include the character's
                # actions into a dataframe

if not combined:
    train_dict = load_obj("train_dictionary")
    test_dict = load_obj("test_dictionary")
    col_names = ['dialogue', 'character']
    train_df = pd.DataFrame(columns=col_names)
    test_df = pd.DataFrame(columns=col_names)

    for key in train_dict:
        for dialogue in train_dict[key]:
            df = pd.DataFrame([[dialogue, key]], columns=col_names)
            train_df = train_df.append(df)
    for key in test_dict:
        for dialogue in test_dict[key]:
            df = pd.DataFrame([[dialogue, key]], columns=col_names)
            test_df = test_df.append(df)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    save_obj(train_df, "train_df")
    save_obj(test_df, "test_df")
else:
    train_dict = load_obj("train_combined")
    test_dict = load_obj("test_combined")
    col_names = ['dialogue', 'action', 'character']
    train_combined_df = pd.DataFrame(columns=col_names)
    test_combined_df = pd.DataFrame(columns=col_names)

    for key in train_dict:
        for dialogue_action_tup in train_dict[key]:
            df = pd.DataFrame([[dialogue_action_tup[0], dialogue_action_tup[1], key]], columns=col_names)
            train_combined_df = train_combined_df.append(df)
    for key in test_dict:
        for dialogue_action_tup in test_dict[key]:
            df = pd.DataFrame([[dialogue_action_tup[0], dialogue_action_tup[1], key]], columns=col_names)
            test_combined_df = test_combined_df.append(df)
    train_combined_df = train_combined_df.reset_index(drop=True)
    test_combined_df = test_combined_df.reset_index(drop=True)

    save_obj(train_combined_df, "train_combined_df")
    save_obj(test_combined_df, "test_combined_df")
