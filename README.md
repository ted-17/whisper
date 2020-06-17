# whisper Conversion

## -Convert the voice into whisper, using LPC analysis-

### 1. Vocal Tract Estimation
Assuming that speech signal is the combination of vocal cords and vocal tract.
<img width="743" alt="speech_generation_model" src="https://user-images.githubusercontent.com/66860222/84912839-cf9e7a80-b0f4-11ea-999f-4aebae83b996.png">

Vocal cords express the source signal, which determine the pitch.
Vocal tract is viewed as auto regressive model, its coefficients(LPC) models the concatenation of vocal tube.


<img width="237" alt="vocal_tube" src="https://user-images.githubusercontent.com/66860222/84912802-c6151280-b0f4-11ea-8b2d-6771198b3cda.png">

It is called Souce-Filter Model.
Then, Estimating LPC, we can extract the characteristics of the vocal tract.

### 2. Whisper conversion
Just simply filtering whitenoise by the estimated vocal tract makes the spectrum whose vocal cords generate nothing but breath, which makes the sound whispering.
<img width="653" alt="vocal_tract_model" src="https://user-images.githubusercontent.com/66860222/84912865-d4fbc500-b0f4-11ea-8394-b7ec887d4271.png">
