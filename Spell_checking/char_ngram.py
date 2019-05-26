import random

text = "Philosophically, ruminations of the human mind and its processes have been around since the times of the ancient Greeks. In 387 BCE, Plato is known to have suggested that the brain was the seat of the mental processes. In 1637, René Descartes posited that humans are born with innate ideas, and forwarded the idea of mind-body dualism, which would come to be known as substance dualism (essentially the idea that the mind and the body are two separate substances). From that time, major debates ensued through the 19th century regarding whether human thought was solely experiential (empiricism), or included innate knowledge (nativism). Some of those involved in this debate included George Berkeleyand John Locke on the side of empiricism, and Immanuel Kant on the side of nativism. With the philosophical debate continuing, the mid to late 19th century was a critical time in the development of psychology as a scientific discipline. Two discoveries that would later play substantial roles in cognitive psychology were Paul Broca's discovery of the area of the brain largely responsible for language production,[3] and Carl Wernicke's discovery of an area thought to be mostly responsible for comprehension of language. Both areas were subsequently formally named for their founders and disruptions of an individual's language production or comprehension due to trauma or malformation in these areas have come to commonly be known as Broca's aphasia and Wernicke's aphasia. From the 1920s to the 1950s, the main approach to psychology was behaviorism. Initially, its adherents viewed mental events such as thoughts, ideas, attention, and consciousness as unobservables, hence outside the realm of a science of psychology. One pioneer of cognitive psychology, who worked outside the boundaries (both intellectual and geographical) of behaviorism was Jean Piaget. From 1926 to the 1950s and into the 1980s, he studied the thoughts, language, and intelligence of children and adults."

n = 3
ngram = {}

# create n grams
for i in range(len(text)-n):
    gram = text[i:i+n]
    if gram not in ngram.keys():
        ngram[gram]=[]
    ngram[gram].append(text[i+n])

# test
currentgram = text[0:n]
result = currentgram
for i in range(100):
    if currentgram not in ngram.keys():
        break
    possibilities = ngram[currentgram]
    nextItem = possibilities[random.randrange(len(possibilities))]

    result += nextItem
    currentgram = result[len(result)-n:len(result)]

print(result)