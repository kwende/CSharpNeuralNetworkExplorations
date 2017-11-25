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

        public double[] ComputeLayerOutput()
        {
            List<double> output = new List<double>();
            foreach (Neuron neuron in Neurons)
            {
                output.Add(neuron.ComputeOutput());
            }
            return output.ToArray();
        }
    }
}
