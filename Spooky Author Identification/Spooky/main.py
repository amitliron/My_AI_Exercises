


if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document, what is the title of this document?',
    ]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
    X           = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names_out())

    num_of_docs  = X.shape[0]
    num_of_words = X.shape[1]

    for doc_num in range(num_of_docs):

        list_of_top_tfidf = []
        for i in X[doc_num].indices:
            list_of_top_tfidf.append(vectorizer.get_feature_names_out()[i])

        print(f"doc_num: {doc_num}: {list_of_top_tfidf}")

    #     # print(f"doc{doc_num}: {X[doc_num]}")
    #     print(f"doc{doc_num}: {vectorizer.get_feature_names_out()[X[doc_num]]}")
    # #     print(f"Doc: {doc_num}: ")
    # #     for word_num in range(num_of_words):
    # #         #print(x[word_num]
    # #         print(word_num)
    # #     break