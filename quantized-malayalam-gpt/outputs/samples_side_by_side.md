# Samples side by side

Prompt `കേരളത്തിൽ`, seed 0, temperature 0.8, top-k 40, 200 new characters. The same seed means the same random numbers, so variants agree until a quantized distribution first picks a different character.

## FP32 baseline (P2 checkpoint)

കേരളത്തിൽനിന്നും സംഭവം കഴിഞ്ഞ ആരോപണം യുവാഗ്രഹിക്കുന്നതും ഒന്നും സുമങ്കാട്ടിലും സ്ഥാനത്ത് വരുന്നു. ഈ വീട്ടിന്റെ സ്ഥാപനത്തിലേക്ക് നല്കുവെന്ന സയ്യം നല്കി തൊട്ടിക്കാനാകുന്ന വാങ്ങിയത്. നിലവില് രാഷ്ട്രീയത്ത് സി വിട്

_English translation: TODO_

## Dynamic int8 (identical to FP32 for the first 24 characters)

കേരളത്തിൽനിന്നും സംഭവം കാട്ട് ആരോപണം യുവാഗ്രഹിക്കുന്നത്. ഒന്നാം സുമങ്കാട്ടില് 19 പാസത്തിന്റെ രൂപതം തിരിച്ചടിയില് ക്ലോമിനറി പരിചരിപ്പിക്കാന് യുത്ത അഞ്ചായിരുന്നു. ഇതാണ് ഈ വാങ്ങിയത്. നിലവില് രാഷ്ട്രീയത്ത് സി വിട്

_English translation: TODO_

## Static int8, calibrated on real text (identical to FP32 for the first 71 characters)

കേരളത്തിൽനിന്നും സംഭവം കഴിഞ്ഞ ആരോപണം യുവാഗ്രഹിക്കുന്നതും ഒന്നും സുമങ്കാരിക നിക്ഷേപകരണ കുറ്റത്തെ തുടർന്ന് മനസിലാക്കിയത്.
റസീലക്ഷ്യൽ രാവിലെ പ്രദേശി അഞ്ചാൽ തന്നെ ഭാഗത്തിൽ നിന്ന് കൊണ്ട് കിടക്കറ്റ് ലഭിച്ച് വിളിച്ചു

_English translation: TODO_

## Static int8, calibrated on torch.randn (identical to FP32 for the first 24 characters)

കേരളത്തിൽനിന്നും സംഭവം കാട്ട് ആരോപണം യുവാഗ്രഹിക്കുന്നതു തിരുവനന്തപുരം ഉടൻ പറഞ്ഞു പറഞ്ഞു. ഇരുപത്തിൽ തിരിച്ചടിയില്ലാത്തത് റിപ്പോർട്ട് ചെയ്യുന്ന സംഭീകരണം തെരഞ്ഞെടുപ്പാവുകയും അത് കൊണ്ട് കിടക്കറ്റ് ലഭിച്ചിരിക്കുക.

_English translation: TODO_

## QAT int8 (300-step fake-quant fine-tune) (identical to FP32 for the first 24 characters)

കേരളത്തിൽനിന്നും സംഭവം കുടുംബശ്രീയ ഏജൻ. ബഗാനിലെ പത്ര പല തിന്റെ വിജയങ്ങൾ ഇവരുടെ സ്ഥാനത്ത് വരുമ്പോൾ കുറുത്തുന്ന ജസ്റ്റ് ചെറിയ കോടതി പരാതി പറഞ്ഞിരുന്ന ചിത്രത്തിൽ ഭാഗമായ മത്സരങ്ങൾക്ക് നടത്തിയിരുന്നു. എതിര്ത്ത് കുറ

_English translation: TODO_

## Control: FP32, same 300-step fine-tune (identical to FP32 for the first 24 characters)

കേരളത്തിൽനിന്നും സംഭവം കുടുംബശ്രീയ ഏജൻ. ബഗാനിലെ പത്ര പല തിന്റെ വിജയങ്ങൾ ഇവരുടെ സ്ഥാനത്ത് വരുമ്പോൾ കുടുംബത്തെ കുടുംബത്തിന് പ്രതിപക്ഷ ചെയ്യുന്ന സംഭീകരണം തെരഞ്ഞെടുപ്പിന്റെ വാങ്ങിയത്. നിലയില് രാഷ്ട്രീയത്തിന് അവകാശ

_English translation: TODO_
