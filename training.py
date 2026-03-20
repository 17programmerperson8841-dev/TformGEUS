import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
layers = tf.keras.layers
pronouns=np.array([
  "person", "man", "woman", "boy", "girl", "child", "adult", "individual", "human", "being",
  "soul", "stranger", "visitor", "guest", "neighbor", "citizen", "peer", "partner", "friend", "companion",
  "leader", "follower", "master", "servant", "expert", "novice", "veteran", "genius", "scholar", "teacher",
  "student", "worker", "artist", "creator", "pioneer", "visionary", "mentor", "protege", "witness", "observer",
  "entity", "presence", "figure", "character", "subject", "specimen", "anomaly", "identity", "essence", "spirit",
  "creature", "beast", "monster", "brute", "fiend", "ghoul", "wraith", "phantom", "specter", "ghost",
  "cyclops", "titan", "colossus", "behemoth", "giant", "dwarf", "goblin", "orc", "elf", "fey",
  "demon", "devil", "angel", "seraph", "cherub", "deity", "god", "goddess", "immortal", "avatar",
  "abomination", "mutant", "hybrid", "chimera", "automaton", "android", "cyborg", "drone", "construct", "golem",
  "shapeshifter", "doppelganger", "mimic", "outcast", "hermit", "wanderer", "nomad", "drifter", "legend", "myth", "johny"
], dtype=object)
#i is index p is position
def ConcatenateArrays(adjO, adjT, pro):
    outNP = np.empty(len(adjO) * len(adjT) * len(pro), dtype=object)
    curr = 0
    for i, p in enumerate(adjO):
        for i2, p2 in enumerate(adjT):
            for i3, p3 in enumerate(pro):
                concatenated = p + " " + p2 + p3
                outNP[curr] = concatenated
                curr += 1
    return outNP
greatAdj1=np.array([
   "A","Kind","A","Brave","A","Sweet","A","Honest",
   "A","Helpful","A","Good","A","Nice","A","Pure",
   "A","Gentle","A","Caring","A","Loyal","A","Wise",
   "A","Lucky","A","Best","A","Happy","A","Smiling",
   "An","Amazing","An","Awesome","An","Incredible","An","Outstanding",
   "An","Exceptional","An","Extraordinary","An","Unstoppable","An","Unbelievable",
   "An","Ultimate","An","Epic","A","Very","Kind","A","Very","Brave",
   "A","Super","Sweet","A","Deeply","Honest","A","Truly","Helpful",
   "A","Quite","Good","A","Completely","Pure","A","Genuinely","Gentle",
   "A","Deeply","Caring","A","Fiercely","Loyal","A","Highly","Wise",
   "A","Unusually","Lucky","A","Totally","Best","A","Extremely","Happy",
   "A","Always","Smiling","A","Truly","Noble","A","Openly","Friendly",
   "An","Absolutely","Perfect","A","Gracefully","Kind","A","Deeply","Calm",
   "A","Completely","Fair","A","Totally","Wonderful","A","Truly","Grand",
   "A","Really","Lovely","A","Extremely","Bold","A","Sharply","Bright",
   "An","Exceptionally","Clever","A","Wildly","Daring","A","Highly","Eager",
   "A","Totally","Fearless","A","Deeply","Generous","A","Truly","Heroic",
   "A","Highly","Inventive","A","Always","Joyful","A","Super","Keen",
   "A","Very","Lively","An","Extremely","Mighty","A","Deeply","Optimistic",
   "An","Endlessly","Patient","A","Surprisingly","Quick","An","Always","Reliable",
   "A","Deeply","Sincere","A","Truly","Thoughtful","A","Completely","Unique",
], dtype=object).flatten()



greatAdj2=np.array([
   " Fiercely","Valiant","","Warmly","Kind","","Deeply","Xenial","",
   " Forever","Youthful","","Absolutely","Zealous","","Extremely","Amazing","",
   " Absolutely","Awesome","","Incredibly","Incredible","","Outstandingly","Brilliant","",
   " Exceptionanolly","Rare","","Extraordinarily","Great","","Unstoppably","Strong","",
   " Unbelievably","Good","","Ultimately","Perfect","","Epically","Heroic","",
   " Truly","Remarkable","","Deeply","Inspiring","","Highly","Admirable","",
   " Completely","Stunning","","Absolutely","Fantastic","","Really","Phenomenal","",
   " Extremely","Impressive","","Deeply","Respectable","","Totally","Magnificent","",
   " Unusually","Spectacular","","Profoundly","Brilliant","","Surprisingly","Excellent","",
   " Deeply","Marvelous","","Truly","Glorious","","Exceptionally","Superb","",
   " Unbelievably","Fantastic","","Totally","Radiant","","Deeply","Splendid","",
   " Highly","Illustrious","","Absolutely","Majestic","","Really","Very","Kind",
   " Extremely","Brave","Man","Deeply","Kind","Spirit","Highly","Generous","Heart",
   " Truly","Outstanding","Soul","Extremely","Loyal","Friend","Unusually","Wise","Elder",
   " Remarkably","Happy","Child","Exceptionally","Caring","Parent","Really","Inspiring","Teacher",
   " Deeply","Thoughtful","Leader","Highly","Patient","Guide","Completely","Reliable","Worker",
   " Profoundly","Honest","Citizen","Really","Friendly","Neighbor","Absolutely","Helpful","Volunteer",
   " Extremely","Smart","Student","Truly","Creative","Artist","Completely","Bold","Designer",
   " Highly","Optimistic","Thinker","Really","Very","Kind","Soul","Deeply","Extremely","Brave",
   " Truly","Incredibly","Kind"
],dtype=object).flatten()



# 75 groups, 2-word each
evilAdj1 = np.array([
"A","Mean","A","Cruel","A","Scary","The","Dark","A","Wicked",
"A","Sneaky","A","Rude","A","Bad","The","Hateful","A","Bitter",
"The","Cold","A","Violent","A","Greedy","The","Sly","A","Savage",
"The","Demonic","A","Heartless","The","Selfish","A","Rotten","The","Nasty",
"A","Spooky","The","Creepy","A","Foul","A","Evil","The","Villainous",
"A","Malevolent","The","Sinister","A","Vicious","A","Devious","The","Ruthless",
"A","Fierce","A","Savage","The","Monstrous","A","Vile","A","Odious",
"The","Nefarious","A","Fiendish","A","Villainous","The","Malicious","A","Perfidious",
"A","Cruelly","The","Treacherous","A","Ghastly","A","Horrendous","The","Atrocious",
"A","Baleful","The","Bloodthirsty","A","Cunning","A","Diabolical","The","Terrible",
"A","Appalling","An","Awful","A","Grim","The","Fearsome","A","Ruthless",
"A","Cruel","The","Maleficent","A","Sinister","A","Vicious","The","Frightful",
"A","Perilous","An","Odious","A","Darkest","The","Villainous","A","Terrible"
], dtype=object).flatten()

# 75 groups, 3-word each
evilAdj2 = np.array([
" Fiercely","Mean","Extremely","Cruel","Terribly","Scary","Darkly","Dark",
"Wickedly","Wicked","Sneakily","Sneaky","Rudely","Rude","Badly","Bad",
" Hateful","Hateful","Bitterly","Bitter","Coldly","Cold","Violently","Violent",
"Greedily","Greedy","Slyly","Sly","Savage","Savage","Demonically","Demonic",
"Heartlessly","Heartless","Selfishly","Selfish","Rottenly","Rotten","Nastily","Nasty",
"Spookily","Spooky","Creepily","Creepy","Foully","Foul","Evilly","Evil",
" Villainously","Villainous","Malevolently","Malevolent","Sinisterly","Sinister","Viciously","Vicious",
"Deviously","Devious","Ruthlessly","Ruthless","Fiercely","Fierce","Savagely","Savage",
" Monstrously","Monstrous","Vilely","Vile","Odiously","Odious","Nefariously","Nefarious",
"Fiendishly","Fiendish","Villainously","Villainous","Maliciously","Malicious","Perfidiously","Perfidious",
"Cruelly","Cruelly","Treacherously","Treacherous","Ghastly","Ghastly","Horrendously","Horrendous",
" Atrociously","Atrocious","Balefully","Baleful","Bloodthirstily","Bloodthirsty","Cunningly","Cunning",
"Diabolically","Diabolical","Terribly","Terrible","Appallingly","Appalling","An","Awfully","Awful",
"Grimly","Grim","Fearsomely","Fearsome","Ruthlessly","Ruthless","Cruelly","Cruel",
" Maleficently","Maleficent","Sinisterly","Sinister","Viciously","Vicious","Frightfully","Frightful",
"Perilously","Perilous","An","Odiously","Odious","Darkly","Dark","Villainously","Villainous",
"Terribly","Terrible"
], dtype=object).flatten()

# 75 groups, 2-word each
uglyAdj1 = np.array([
"A","Unpleasant","A","Hideous","A","Ghastly","The","Grotesque","A","Repulsive",
"A","Unattractive","A","Foul","The","Disgusting","A","Awful","The","Revolting",
"A","Distasteful","The","Offensive","A","Monstrous","The","Grim","A","Nasty",
"The","Loathsome","A","Horrid","The","Odious","A","Vile","The","Repugnant",
"A","Ugly","The","Detestable","A","Frightful","The","Abominable","A","Disagreeable",
"A","Horrible","The","Pathetic","A","Unseemly","The","Appalling","A","Shocking",
"An","Unlovely","A","Unsightly","The","Unpleasantest","A","Displeasing","A","Horrifying",
"The","Frightful","A","Grotesque","An","Abhorrent","A","Repelling","The","Ugliest",
"A","Gruesome","A","Hideously","The","Unattractive","A","Disgusting","A","Nasty",
"The","Revolting","A","Odious","The","Monstrous","A","Loathsome","The","Abhorrent",
"A","Unpleasant","A","Uglyest","The","Foul","A","Awful","The","Detestable",
"A","Horrendous","An","Hideous","A","Ghastly"
], dtype=object).flatten()

# 75 groups, 3-word each
uglyAdj2 = np.array([
" Unpleasantly","Unpleasant","Hideously","Hideous","Ghastly","Ghastly","Grotesquely","Grotesque",
" Repulsively","Repulsive","Unattractively","Unattractive","Foully","Foul","Disgustingly","Disgusting",
" Awfully","Awful","Revoltingly","Revolting","Distastefully","Distasteful","Offensively","Offensive",
" Monstrously","Monstrous","Grimly","Grim","Nastily","Nasty","Loathsomely","Loathsome",
" Horridly","Horrid","Odiously","Odious","Vilely","Vile","Repugnantly","Repugnant",
" Uglyly","Ugly","Detestably","Detestable","Frightfully","Frightful","Abominably","Abominable",
" Disagreeably","Disagreeable","Horribly","Horrible","Pathetically","Pathetic","Unseemly","Unseemly",
" Appallingly","Appalling","Shockingly","Shocking","Unlovely","Unlovely","Unsightly","Unsightly",
" Unpleasantest","Unpleasantest","Displeasingly","Displeasing","Horrifyingly","Horrifying","Frightfully","Frightful",
" Grotesquely","Grotesque","Abhorrently","Abhorrent","Repellingly","Repelling","Ugliest","Ugliest",
" Gruesomely","Gruesome","Hideously","Hideous","Unattractively","Unattractive","Disgustingly","Disgusting",
" Nastily","Nasty","Revoltingly","Revolting","Odiously","Odious","Monstrously","Monstrous",
" Loathsomely","Loathsome","Abhorrently","Abhorrent","Unpleasantly","Unpleasant","Uglyest","Uglyest",
" Foully","Foul","Awfully","Awful","Detestably","Detestable","Horrendously","Horrendous",
" Hideously","Hideous","Ghastly","Ghastly"
], dtype=object).flatten()

# 75 groups, 2-word each
shinyAdj1 = np.array([
"A","Bright","A","Gleaming","A","Glittering","The","Radiant","A","Lustrous",
"A","Polished","A","Glossy","The","Shimmering","A","Sparkling","The","Dazzling",
"A","Glorious","The","Twinkling","A","Luminous","The","Beaming","A","Glistening",
"The","Shiny","A","Glittery","The","Glossy","A","Brilliant","A","Shimmering",
"The","Lustrous","A","Sparkling","A","Dazzling","The","Radiant","A","Brightest",
"A","Glittery","The","Shiniest","A","Polished","A","Shinyest","The","Glistening",
"A","Glorious","A","Luminous","The","Dazzling","A","Twinkling","An","Beaming",
"A","Radiant","The","Sparkling","A","Glittering","A","Brilliant","The","Lustrous",
"A","Twinkling","The","Shimmering","A","Brightly","A","Glorious","The","Glistening",
"A","Beaming","A","Shiny","The","Brilliant","A","Polished","An","Radiant"
], dtype=object).flatten()

# 75 groups, 3-word each
shinyAdj2 = np.array([
" Brilliantly","Bright","Gleamingly","Gleaming","Glitteringly","Glittering","Radiantly","Radiant",
" Lustrously","Lustrous","Polishedly","Polished","Shimmeringly","Shimmering","Sparklingly","Sparkling",
" Dazzlingly","Dazzling","Gloriously","Glorious","Twinklingly","Twinkling","Luminously","Luminous",
" Beamingly","Beaming","Glisteningly","Glistening","Shinily","Shiny","Glitterily","Glittery",
" Glossily","Glossy","Brilliantly","Brilliant","Shimmeringly","Shimmering","Lustrously","Lustrous",
" Sparklingly","Sparkling","Dazzlingly","Dazzling","Radiantly","Radiant","Brightly","Bright",
" Glitterily","Glittery","Shiniestly","Shiniest","Polishedly","Polished","Shinyest","Shinyest",
" Glisteningly","Glistening","Gloriously","Glorious","Luminously","Luminous","Dazzlingly","Dazzling",
" Twinklingly","Twinkling","Shimmeringly","Shimmering","Brightly","Bright","Gloriously","Glorious",
" Glisteningly","Glistening","Beamingly","Beaming","Shinily","Shiny","Brilliantly","Brilliant",
" Polishedly","Polished","Radiantly","Radiant"
], dtype=object).flatten()
great = ConcatenateArrays(greatAdj1, greatAdj2, pronouns)
evil = ConcatenateArrays(evilAdj1, evilAdj2, pronouns)
ugly = ConcatenateArrays(uglyAdj1, uglyAdj2, pronouns)
shiny = ConcatenateArrays(shinyAdj1, shinyAdj2, pronouns)

data = np.concatenate([
    great, evil, ugly, shiny
], dtype=object)
#0 MEANS GREAT 1 MEANS EVIL 2 MEANS UGLY AND 3 MEANS SHINY
labels = np.concatenate([
    np.zeros(len(great)),
    np.ones(len(evil)),
    np.full(len(ugly), 2.0),
    np.full(len(shiny), 3.0)
], dtype=float)
indices = np.arange(len(data))


np.random.seed(42) 
np.random.shuffle(indices)
data = data[indices]
labels= labels[indices]
max_tokens = 250000
max_len = 20

processor = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len
)
processor.adapt(data)
new_data = processor(data)

inputs = layers.Input(shape=(max_len,), dtype = 'int64')
p = layers.Embedding(input_dim=max_tokens, output_dim=25)(inputs)

positions = tf.range(start=0, limit=max_len, delta=1)
encoding = layers.Embedding(input_dim=max_len, output_dim=25)(positions)
p = p + encoding

#ATTENTION ATTENTION (stable_p is such a cool vector)
stable_p = layers.LayerNormalization(axis=[2])(p)
attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=16)(stable_p, stable_p)
p = layers.Add()([attention_output, p])     #wow it adds stuff

#THE SUPER COOL PART WOWOWOWOW
stable_p = layers.LayerNormalization(axis=[2])(p)
mlp_output = layers.Dense(25, activation='relu')(stable_p)
p = layers.Add()([mlp_output, p])
p = layers.Dropout(0.1)(p)

#ATTENTION ATTENTION (stable_p is such a cool vector)
stable_p = layers.LayerNormalization(axis=[2])(p)
attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=16)(stable_p, stable_p)
p = layers.Add()([attention_output, p])     #wow it adds stuff

#THE SUPER COOL PART WOWOWOWOW
stable_p = layers.LayerNormalization(axis=[2])(p)
mlp_output = layers.Dense(25, activation='relu')(stable_p)
p = layers.Add()([mlp_output, p])
p = layers.Dropout(0.1)(p)

#ATTENTION ATTENTION (stable_p is such a cool vector)
stable_p = layers.LayerNormalization(axis=[2])(p)
attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=16)(stable_p, stable_p)
p = layers.Add()([attention_output, p])     #wow it adds stuff

#THE SUPER COOL PART WOWOWOWOW
stable_p = layers.LayerNormalization(axis=[2])(p)
mlp_output = layers.Dense(25, activation='relu')(stable_p)
p = layers.Add()([mlp_output, p])
p = layers.Dropout(0.1)(p)

#ATTENTION ATTENTION (stable_p is such a cool vector)
stable_p = layers.LayerNormalization(axis=[2])(p)
attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=16)(stable_p, stable_p)
p = layers.Add()([attention_output, p])     #wow it adds stuff

#THE SUPER COOL PART WOWOWOWOW
stable_p = layers.LayerNormalization(axis=[2])(p)
mlp_output = layers.Dense(25, activation='relu')(stable_p)
p = layers.Add()([mlp_output, p])
p = layers.Dropout(0.1)(p)

#ER IT JUST MAKES EVERYTHING BETTER AND USEFUL
p = layers.GlobalAveragePooling1D()(p)
outputs = layers.Dense(4, activation='softmax')(p)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

preTrain = tf.data.Dataset.from_tensor_slices((new_data, labels))
preTrain = preTrain.shuffle(10000).batch(2028).prefetch(tf.data.AUTOTUNE)
#WOAH IT DOES STUFF
model.fit(preTrain, epochs=5, verbose=1)
testStuff = np.array(["A beautiful man named Johny", "A mean Johny", "A guy who throws things at people"], dtype=object)

vectorStuff = processor(testStuff)
predictions = np.argmax(model.predict(vectorStuff, verbose=0), axis=1)

filter = {0: "GREAT", 1: "EVIL", 2: "UGLY", 3: "SHINY"}
for inp, outp in enumerate(predictions):
    print(f"DESCRIPTION: {testStuff[inp]} HAS BEEN JUDGED AS {filter[outp]}")


inference_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="text_input")
x = processor(inference_input)
output = model(x)
full_inference_model = tf.keras.Model(inference_input, output)
full_inference_model.export("TformGEUS")