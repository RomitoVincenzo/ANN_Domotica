using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using System;

namespace AI_Domotic
{
    class Program
    {

        /* Declares jagged input array */
        public static double[][] SensoriInput = {
            new[] {0.0, 0.0, 0.0, 0.0},
            new[] {0.0, 0.0, 0.0, 1.0},
            new[] {0.0, 0.0, 1.0, 0.0},
            new[] {0.0, 0.0, 1.0, 1.0},
            new[] {0.0, 1.0, 0.0, 0.0},
            new[] {0.0, 1.0, 0.0, 1.0},
            new[] {0.0, 1.0, 1.0, 0.0},
            new[] {0.0, 1.0, 1.0, 1.0},
            new[] {1.0, 0.0, 0.0, 0.0},
            new[] {1.0, 0.0, 0.0, 1.0},
            new[] {1.0, 0.0, 1.0, 0.0},
            new[] {1.0, 0.0, 1.0, 1.0},
            new[] {1.0, 1.0, 0.0, 0.0},
            new[] {1.0, 1.0, 0.0, 1.0},
            new[] {1.0, 1.0, 1.0, 0.0},
            new[] {1.0, 1.0, 1.0, 1.0},
        };

        /* Declares jagged input array */
        public static double[][] AttuatoriOutput = {
            new[] {1.0, 1.0, 0.0, 0.0, 0.0},
            new[] {1.0, 1.0, 0.0, 0.0, 0.0},
            new[] {1.0, 1.0, 0.0, 0.0, 0.0},
            new[] {1.0, 1.0, 0.0, 0.0, 0.0},
            new[] {1.0, 0.0, 0.0, 0.0, 0.0},
            new[] {1.0, 0.0, 0.0, 0.0, 0.0},
            new[] {1.0, 0.0, 0.0, 0.0, 0.0},
            new[] {1.0, 0.0, 0.0, 0.0, 0.0},
            new[] {0.0, 1.0, 1.0, 1.0, 1.0},
            new[] {0.0, 1.0, 1.0, 0.0, 0.0},
            new[] {0.0, 1.0, 0.0, 1.0, 1.0},
            new[] {0.0, 1.0, 0.0, 0.0, 0.0},
            new[] {0.0, 0.0, 1.0, 1.0, 1.0},
            new[] {0.0, 0.0, 1.0, 0.0, 0.0},
            new[] {0.0, 0.0, 0.0, 1.0, 1.0},
            new[] {0.0, 0.0, 0.0, 0.0, 0.0},
        };


        static void Main(string[] args)
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 4));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 4));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
            network.Structure.FinalizeStructure();
            network.Reset();
            IMLDataSet trainingSet = new BasicMLDataSet(SensoriInput, AttuatoriOutput);
            IMLTrain train = new ResilientPropagation(network, trainingSet);
            int epoch = 1;
            do
            {
                /* Avvia la redistribuzione dei pesi */
                train.Iteration();
                Console.WriteLine(@"Epoch #" + epoch + @" Error:" + train.Error);
                epoch++;
            } while (train.Error > 0.001); /* Itera finche' non viene raggiunto un errore tollerabile */

            /* Test la MLP */
            Console.WriteLine("\r\n+------------------------------------+");
            Console.WriteLine("|Neural Network Results:             |");
            Console.WriteLine("+------------------------------------+");
            foreach (IMLDataPair pair in trainingSet)
            {
                IMLData output = network.Compute(pair.Input);
                Console.WriteLine("Input:" + pair.Input[0] + " - " + pair.Input[1] + " - " + pair.Input[2] + " - " + pair.Input[3]
                                  + "\tactual=" + Math.Round(output[0], 2) + " - " + Math.Round(output[1], 2) + " - " + Math.Round(output[2], 2) + " - " + Math.Round(output[3], 2) + " - " + Math.Round(output[4], 2)
                                  + "\tideal=" + pair.Ideal[0] + " - " + pair.Ideal[1] + " - " + pair.Ideal[2] + " - " + pair.Ideal[3] + " - " + pair.Ideal[4]);
            }
            Console.Read();
        }
    }
}
