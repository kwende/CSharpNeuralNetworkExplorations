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
        private double _sizeOfTrainingSet = 0.0;
        public L1Normalization(int sizeOfTrainingSet, double regularizationConstant)
        {
            _regularizationConstant = regularizationConstant;
            _sizeOfTrainingSet = sizeOfTrainingSet;
        }

        public double Compute(double weight)
        {
            return (System.Math.Sign(weight) / (_sizeOfTrainingSet)) * _regularizationConstant;
        }
    }
}
