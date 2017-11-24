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
        public Layer()
        {
            Neurons = new List<Neuron>();
        }

        public double[] ComputeFullLayerOutputConsideringDropouts(double probabilityOfDropout)
        {
            List<double> output = new List<double>();
            foreach (Neuron neuron in Neurons)
            {
                output.Add(neuron.ComputeOutput(probabilityOfDropout));
            }
            return output.ToArray();
        }

        public double[] ComputeFullLayerOutput()
        {
            List<double> output = new List<double>();
            foreach (Neuron neuron in Neurons)
            {
                output.Add(neuron.ComputeOutput(0));
            }
            return output.ToArray();
        }

        public double[] ComputeLayerOutputWithDropouts(double probabilityOfDropout, Random rand)
        {
            List<double> output = new List<double>();
            foreach (Neuron currentLayerNeuron in Neurons)
            {
                if (probabilityOfDropout >= rand.NextDouble())
                {
                    currentLayerNeuron.DropOut = true;
                }
                else
                {
                    currentLayerNeuron.DropOut = false;
                    output.Add(currentLayerNeuron.ComputeOutput());
                }
            }
            return output.ToArray();
        }
    }
}
