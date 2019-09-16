using System;
using MLappML.Model.DataModels;
using Microsoft.ML;

namespace MLapp
{
    class Program
    {
        static void Main(string[] args)
        {
            //load teh model
            MLContext mlContext = new MLContext();

            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out var modelInputSchema);

            var predEngin = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            //Creating the input data
            ModelInput input = new ModelInput();
            while (true) { 
            input.SentimentText = Console.ReadLine();

            // Sample test- result
            ModelOutput result = predEngin.Predict(input);

            Console.WriteLine($"Text: {input.SentimentText} | Prediction: {(Convert.ToBoolean(result.Prediction) ? "Toxic" : "Non Toxic")} sentiment");
            if (Console.Read() == 32)
                break;
            }


        }
    }
}
