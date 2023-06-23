# TranslatorV2

This is an implementation of an English-French translator based on a decoder-only transformer model. </br>
Different to EngFr-Translator V1 as it is a decoder-only architecture. </br>
The model translates to French with the following prompt:
```
<|start|>Example English sentence here...<|equals|>
```
Having been trained on 10GB of data in the format:
```
<|start|>English...<|equals|>French...<|endoftext|><|start|>English2...<|equals|>...
```