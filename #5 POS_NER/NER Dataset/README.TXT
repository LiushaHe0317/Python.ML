
Description
Context: Annotated Corpus for Named Entity Recognition using GMB(Groningen Meaning Bank) corpus for entity classification with enhanced and popular features by Natural Language Processing applied to the data set.
Please refer to Column Metadata for better illustrations. Please note the following: start1 and end1 tags are to demonstrate the prefix and suffix of start and end of a line.
For example:
How is he
Previous word of How will become start1 and next word of he will become end1
Integers(1 & 2) in start or end tags are for previous and previous-to-previous or next and next-to-next respectively.

Columns
    lemmaLemma of a token in sentence
    next-lemmaLemma of next token in sentence
    next-next-lemmaLemma of token at +2nd position to the current token in sentence
    next-next-posPOS tag of token at +2nd position to the current token in sentence
    next-next-shapeShape of token at +2nd position to the current token in sentence
    next-next-wordToken at +2nd position to the current token in sentence
    next-posPOS tag of the next(+1 position) token
    next-shapeShape of the next(+1 position) token
    next-wordNext(+1 position) token
    posPOS tag of current token
    prev-iobIOB annotation of previous token
    prev-lemmaLemma of previous token
    prev-posPOS tag for previous token
    prev-prev-iobIOB annotation of token at -2nd position to the current token in sentence
    prev-prev-lemmaLemma of token at -2nd position to the current token in sentence
    prev-prev-posPOS tag of token at -2nd position to the current token in sentence
    prev-prev-shapeShape of token at -2nd position to the current token in sentence
    prev-prev-wordToken at -2nd position to the current token in sentence
    prev-shapeShape of previous(-1 position to current token) token
    prev-wordPrevious word(-1 position to current token)
    sentence_idxSentence Index(Tokens having same index belongs to same sentence)
    shapeShape of the token in sentence
    wordOften termed as Token
    tagIOB annotation of current token; what is this?
