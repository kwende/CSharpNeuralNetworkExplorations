using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    [Serializable]
    public abstract class Layer
    {
        public List<Neuron> Neurons { get; set; }
        public List<int> DropOutMask { get; set; }
        public bool IsDropoutLayer
        {
            get
            {
                return _probabilityOfDropout > 0.0;
            }
        }

        private double _probabilityOfDropout;
        private Random _random;

        public Layer(int numberOfNeurons, double probabilityOfDropout, Random rand)
        {
            Neurons = new List<Neuron>(numberOfNeurons);
            DropOutMask = new List<int>(numberOfNeurons);
            for (int c = 0; c < numberOfNeurons; c++)
            {
                DropOutMask.Add(1);
            }
            _probabilityOfDropout = probabilityOfDropout;
            _random = rand;
        }

        public void UpdateDropoutMask()
        {
            DropOutMask.Clear();
            foreach (Neuron neuron in Neurons)
            {
                if (_random.NextDouble() <= _probabilityOfDropout)
                {
                    DropOutMask.Add(0);
                }
                else
                {
                    DropOutMask.Add(1);
                }
            }
        }

        public double[] ComputeLayerExecutionOutput()
        {
            List<double> output = new List<double>();
            for (int n = 0; n < Neurons.Count; n++)
            {
                Neuron neuron = Neurons[n];

                output.Add(neuron.ComputeExecutionOutput(_probabilityOfDropout));
            }
            return output.ToArray();
        }

        public double[] ComputeLayerTrainingOutput()
        {
            List<double> output = new List<double>();

            for (int n = 0; n < Neurons.Count; n++)
            {
                Neuron neuron = Neurons[n];
                double dropOutBit = DropOutMask[n];

                output.Add(neuron.ComputeTrainingOutput(dropOutBit));
            }
            return output.ToArray();
        }
    }
}
