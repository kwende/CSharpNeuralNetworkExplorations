using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Math.RegularizationFunctions
{
    public class L1Normalization : IRegularizationFunction
    {
        private double _regularizationConstant = 0.0;
        public L1Normalization(double regularizationConstant)
        {
            _regularizationConstant = regularizationConstant;
        }

        public double Compute(double weight, int sizeOfTrainingSet)
        {
            return (System.Math.Sign(weight) / (sizeOfTrainingSet * 1.0)) * _regularizationConstant;
        }
    }
}
