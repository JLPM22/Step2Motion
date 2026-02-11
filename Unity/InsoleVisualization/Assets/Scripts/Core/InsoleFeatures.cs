using System;
using System.IO;
using Unity.Mathematics;
using UnityEngine;

public class InsoleFeatures
{
    public float MaxTotalForce { get; private set; }
    public float MaxTotalPressure { get; private set; }
    public float MaxOneComponentAcceleration { get; private set; }

    private FeaturesJSON FeaturesInternal;

    public void ImportFromJSON(string jsonPath)
    {
        string json = File.ReadAllText(jsonPath);
        FeaturesInternal = JsonUtility.FromJson<FeaturesJSON>(json);

        { 
            int onePercent = Mathf.Max(1, Mathf.RoundToInt(FeaturesInternal.l_total_force.Length * 0.01f));
            float[] copyLForces = new float[FeaturesInternal.l_total_force.Length];
            float[] copyRForces = new float[FeaturesInternal.r_total_force.Length];
            Array.Copy(FeaturesInternal.l_total_force, copyLForces, FeaturesInternal.l_total_force.Length);
            Array.Copy(FeaturesInternal.r_total_force, copyRForces, FeaturesInternal.r_total_force.Length);
            Array.Sort(copyLForces);
            Array.Sort(copyRForces);
            int count = 0;
            float lSum = 0;
            float rSum = 0;
            for (int i = copyLForces.Length - onePercent; i < copyLForces.Length; i++) 
            {
                lSum += copyLForces[i];
                rSum += copyRForces[i];
                count++;
            }
            MaxTotalForce = ((lSum / count) + (rSum / count)) / 2.0f;
        }
        {
            int onePercent = Mathf.Max(1, Mathf.RoundToInt(FeaturesInternal.l_pressures.Length * 0.01f));
            float[] copyLPressures = new float[FeaturesInternal.l_pressures.Length];
            float[] copyRPressures = new float[FeaturesInternal.r_pressures.Length];
            Array.Copy(FeaturesInternal.l_pressures, copyLPressures, FeaturesInternal.l_pressures.Length);
            Array.Copy(FeaturesInternal.r_pressures, copyRPressures, FeaturesInternal.r_pressures.Length);
            Array.Sort(copyLPressures);
            Array.Sort(copyRPressures);
            int count = 0;
            float lSum = 0;
            float rSum = 0;
            for (int i = copyLPressures.Length - onePercent; i < copyLPressures.Length; i++)
            {
                lSum += copyLPressures[i];
                rSum += copyLPressures[i];
                count++;
            }
            MaxTotalPressure = ((lSum / count) + (rSum / count)) / 2.0f;
        }
        {
            int onePercent = Mathf.Max(1, Mathf.RoundToInt(FeaturesInternal.l_accelerations.Length * 0.01f));
            float[] copyLAcc = new float[FeaturesInternal.l_accelerations.Length];
            float[] copyRAcc = new float[FeaturesInternal.r_accelerations.Length];
            Array.Copy(FeaturesInternal.l_accelerations, copyLAcc, FeaturesInternal.l_accelerations.Length);
            Array.Copy(FeaturesInternal.r_accelerations, copyRAcc, FeaturesInternal.r_accelerations.Length);
            Array.Sort(copyLAcc);
            Array.Sort(copyRAcc);
            int count = 0;
            float lSum = 0;
            float rSum = 0;
            for (int i = copyLAcc.Length - onePercent; i < copyLAcc.Length; i++)
            {
                lSum += copyLAcc[i];
                rSum += copyLAcc[i];
                count++;
            }
            MaxOneComponentAcceleration = ((lSum / count) + (rSum / count)) / 2.0f;
        }
    }

    public Features GetFrame(int frame)
    {
        const int nPressureFeatures = 16;
        Features features = new()
        {
            LeftPressure = new float[16],
            RightPressure = new float[16],
            LeftAcceleration = new float3(FeaturesInternal.l_accelerations[frame * 3], FeaturesInternal.l_accelerations[frame * 3 + 1], -FeaturesInternal.l_accelerations[frame * 3 + 2]), // Unity is left-handed, negate Z axis
            RightAcceleration = new float3(FeaturesInternal.r_accelerations[frame * 3], FeaturesInternal.r_accelerations[frame * 3 + 1], -FeaturesInternal.r_accelerations[frame * 3 + 2]),
            LeftAngVelocity = new float3(FeaturesInternal.l_angular_accelerations[frame * 3], FeaturesInternal.l_angular_accelerations[frame * 3 + 1], -FeaturesInternal.l_angular_accelerations[frame * 3 + 2]), // TODO: Unity is left-handed, makes sense negate z axis angular acceleration?
            RightAngVelocity = new float3(FeaturesInternal.r_angular_accelerations[frame * 3], FeaturesInternal.r_angular_accelerations[frame * 3 + 1], -FeaturesInternal.r_angular_accelerations[frame * 3 + 2]),
            LeftTotalForce = FeaturesInternal.l_total_force[frame],
            RightTotalForce = FeaturesInternal.r_total_force[frame],
            LeftCenterOfPressure = new float2(FeaturesInternal.l_center_of_pressure[frame * 2], FeaturesInternal.l_center_of_pressure[frame * 2 + 1]),
            RightCenterOfPressure = new float2(FeaturesInternal.r_center_of_pressure[frame * 2], FeaturesInternal.r_center_of_pressure[frame * 2 + 1]),
        };
        Array.Copy(FeaturesInternal.l_pressures, frame * nPressureFeatures, features.LeftPressure, 0, nPressureFeatures);
        Array.Copy(FeaturesInternal.r_pressures, frame * nPressureFeatures, features.RightPressure, 0, nPressureFeatures);
        return features;
    }

    public struct Features
    {
        public float[] LeftPressure;
        public float[] RightPressure;
        public float3 LeftAcceleration;
        public float3 RightAcceleration;
        public float3 LeftAngVelocity;
        public float3 RightAngVelocity;
        public float LeftTotalForce;
        public float RightTotalForce;
        public float2 LeftCenterOfPressure;
        public float2 RightCenterOfPressure;
    }

    [System.Serializable]
    public struct FeaturesJSON
    {
        public float[] l_pressures;
        public float[] l_accelerations;
        public float[] l_angular_accelerations;
        public float[] l_total_force;
        public float[] l_center_of_pressure;
        public float[] r_pressures;
        public float[] r_accelerations;
        public float[] r_angular_accelerations;
        public float[] r_total_force;
        public float[] r_center_of_pressure;
    }
}
