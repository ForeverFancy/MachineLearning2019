import os
import math


# %%
def train():
    train_files = os.listdir("./hw3-nb/train-mails")

    vocabulary = []
    doc_spam = []
    doc_not_spam = []
    spam_mail_num = 0
    not_spam_mail_num = 0
    for mail_path in train_files:
        if mail_path.startswith("spmsg"):
            spam_mail_num += 1
            with open("./hw3-nb/train-mails/" + mail_path, "r+") as f:
                # Body is the 3th line.
                raw_data = f.readlines()[2].split(" ")
                text = [token for token in raw_data if token.isalpha()]
                vocabulary += text
                doc_spam += text
        else:
            not_spam_mail_num += 1
            with open("./hw3-nb/train-mails/" + mail_path, "r+") as f:
                # Body is the 3th line.
                raw_data = f.readlines()[2].split(" ")
                text = [token for token in raw_data if token.isalpha()]
                vocabulary += text
                doc_not_spam += text
    # print(vocabulary)
    print("Finish scan.")
    vocabulary = list(set(vocabulary))
    vocabulary_num = len(vocabulary)

    print(vocabulary_num)

    P_spam = spam_mail_num / (spam_mail_num + not_spam_mail_num)
    P_not_spam = not_spam_mail_num / (spam_mail_num + not_spam_mail_num)

    nc_spam = len(doc_spam)
    nc_not_spam = len(doc_not_spam)

    P_w_spam = []
    P_w_not_spam = []

    for word in vocabulary:
        nck_spam = doc_spam.count(word)
        nck_not_spam = doc_not_spam.count(word)
        P_w_spam.append((nck_spam + 1) / (nc_spam + vocabulary_num))
        P_w_not_spam.append((nck_not_spam + 1) /
                            (nc_not_spam + vocabulary_num))

    print("Finish Calculate.")

    return vocabulary, P_w_spam, P_w_not_spam, P_spam, P_not_spam


# %%
def test(vocabulary: list, P_w_spam: list, P_w_not_spam: list, P_spam: float, P_not_spam: float):
    num_spam = 0
    num_not_spam = 0
    tp, fn, fp, tn = 0, 0, 0, 0
    count = 0

    test_files = os.listdir("./hw3-nb/test-mails")
    for mail_path in test_files:
        if count % 10 == 0:
            print(count)
        count += 1

        with open("./hw3-nb/test-mails/" + mail_path, "r") as f:
            P_pred_is_spam = 0
            P_pred_not_spam = 0
            raw_data = f.readlines()[2].split(" ")
            text = [token for token in raw_data if token.isalpha()]
            for token in text:
                if token in vocabulary:
                    index = vocabulary.index(token)
                    P_pred_is_spam += math.log(P_w_spam[index])
                    P_pred_not_spam += math.log(P_w_not_spam[index])
            P_pred_is_spam += math.log(P_spam)
            P_pred_not_spam += math.log(P_not_spam)
            if P_pred_is_spam > P_pred_not_spam:
                if mail_path.startswith("spmsg"):
                    tp += 1
                else:
                    fp += 1
            else:
                if mail_path.startswith("spmsg"):
                    fn += 1
                else:
                    tn += 1
    P = tp/(tp+fp)
    R = tp/(tp+fn)
    F1 = 2/(1/P + 1/R)

    print("tp: %d\nfp: %d\nfn: %d\ntn: %d\n" % (tp, fp, fn, tn))
    print("P:", P)
    print("R:", R)
    print("F1 score:", F1)


# %%
if __name__ == "__main__":
    vocabulary, P_w_spam, P_w_not_spam, P_spam, P_not_spam = train()


# %%
    test(vocabulary, P_w_spam, P_w_not_spam, P_spam, P_not_spam)
