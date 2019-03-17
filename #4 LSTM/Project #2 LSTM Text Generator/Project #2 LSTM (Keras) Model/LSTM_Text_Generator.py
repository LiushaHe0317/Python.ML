
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load

def generate_text(
        model, tokenizer, slen, seed_text, num_words):
    
    output_text = []        # define a list of final output 
    input_text = seed_text  # initial seeding text
    
    for i in range(num_words):
        #* transform input text into sequences
        encoded_text = tokenizer.texts_to_sequences(
                [input_text])[0]
        print('encoded text sequences')
        print(encoded_text)
        #* pad the sequence if it is super long or short
        pad_encoded = pad_sequences(
                [encoded_text], 
                maxlen = slen, 
                truncating = 'pre')
        print('the pad sequence')
        print(pad_encoded)
        # generate a predicted word
        pred_word_idx = model.predict_classes(pad_encoded, verbose = 0)[0]
        pred_word = tokenizer.index_word[pred_word_idx]
        
        # update input text
        input_text += ' ' + pred_word
        # append to final output
        output_text.append(pred_word)
    
    return ' '.join(output_text)

model = load_model('nobydick_model.h5')
tokenizer = load(open('nobydick_tokenizer', 'rb'))

# write a sentence
temp = input('Please Give Your Sentence: ')
seed_text = str(temp)

## call the model
pred_text = generate_text(model, tokenizer, slen = 25, seed_text = seed_text, num_words = 25)

print(pred_text)
