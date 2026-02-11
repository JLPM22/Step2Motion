using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "InsoleManagerData", menuName = "InsoleManagerData", order = 1)]
public class InsoleManagerData : ScriptableObject
{
    public string Name;

    [Header("Environment")]
    public float FloorHeight = 0.0f;

    [Header("Paths")]
    public string ModelsPath;
    public string ModelName;
    public int[] Seeds;

    [Header("Skeleton")]
    public Color Color = Color.yellow;
    public string LeftFoot;
    public string LeftToes;
    public string RightFoot;
    public string RightToes;

    [Header("Insole")]
    public Vector3 LeftLocalPosition = new Vector3(-0.052f, -0.041f, -0.114f);
    public Quaternion LeftLocalRotation = new Quaternion(-0.7736382f, -0.02778463f, -0.01684882f, 0.6327938f);
    public Vector3 RightLocalPosition = new Vector3(0.083f, -0.04f, -0.112f);
    public Quaternion RightLocalRotation = new Quaternion(-0.7736382f, -0.02778463f, -0.01684882f, 0.6327938f);
}
