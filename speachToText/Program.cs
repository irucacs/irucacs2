using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;
using static System.Console;
using static System.Environment;
using OpenAI_API;
using OpenAI_API.Completions;

class Program
{
    // Azure Cognitive Speachの "SPEECH_KEY" and "SPEECH_REGION"
    static string speechKey = Environment.GetEnvironmentVariable("SPEECH_KEY");
    static string speechRegion = Environment.GetEnvironmentVariable("SPEECH_REGION");

    //OpenAIのKEY
    static string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");

    static string OutputSpeechRecognitionResult(SpeechRecognitionResult speechRecognitionResult)
    {
        switch (speechRecognitionResult.Reason)
        {
            case ResultReason.RecognizedSpeech:
                //speach to textが成功した場合返却されたテキストを返す
                Console.WriteLine($"RECOGNIZED: Text={speechRecognitionResult.Text}");
                return speechRecognitionResult.Text;
            case ResultReason.NoMatch:
                Console.WriteLine($"NOMATCH: Speech could not be recognized.");
                break;
            case ResultReason.Canceled:
                var cancellation = CancellationDetails.FromResult(speechRecognitionResult);
                Console.WriteLine($"CANCELED: Reason={cancellation.Reason}");

                if (cancellation.Reason == CancellationReason.Error)
                {
                    Console.WriteLine($"CANCELED: ErrorCode={cancellation.ErrorCode}");
                    Console.WriteLine($"CANCELED: ErrorDetails={cancellation.ErrorDetails}");
                    Console.WriteLine($"CANCELED: Did you set the speech resource key and region values?");
                }
                break;     
        }
        return "speachToTextErr";
    }

    async static Task Main(string[] args)
    {
        var speechConfig = SpeechConfig.FromSubscription(speechKey, speechRegion);
        speechConfig.SpeechRecognitionLanguage = "ja-JP";

        using var audioConfig = AudioConfig.FromDefaultMicrophoneInput();
        using var speechRecognizer = new SpeechRecognizer(speechConfig, audioConfig);

        Console.WriteLine("Speak into your microphone.");
        var speechRecognitionResult = await speechRecognizer.RecognizeOnceAsync();
        //音声解析の結果が正常であった場合は対応するテキスト、異常であった場合は異常コード"speachToTextErr"を受領する。
        string getSpeachText = OutputSpeechRecognitionResult(speechRecognitionResult);
        
        //音声解析の結果がエラー出なかった場合ChatGPTAPIを発行する。
        if (getSpeachText != "speachToTextErr")
        {
            //OpenAIAPIオブジェクトを作成
            OpenAIAPI api = new(apiKey);
            //音声入力の結果をプロンプトとして受け取る
            string prompt = getSpeachText;
            Console.WriteLine("PROMPT: "+prompt);
            //プロンプトを発行して回答を受け取る
            string result = await api.Completions.GetCompletion(prompt);
            Console.Write(result.Choices.First().Message.Content);
            Console.ReadLine();
        }
    }
}