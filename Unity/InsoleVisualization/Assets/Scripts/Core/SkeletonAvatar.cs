using BVH;
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class SkeletonAvatar : MonoBehaviour
{
    public float Smoothness = 0.7f;
    public float TransitionSmoothness = 5.0f;
    public float TransitionSmoothnessRoot = 10.0f;
    [Range(0.0f, 1.0f)] public float TransitionDiff = 0.999f;
    [Range(0.0f, 1.0f)] public float TransitionDiffSpine = 1.0f;

    public Transform[] SkeletonTransforms { get; private set; }

    public Transform LeftFoot { get; private set; }
    public Transform LeftToes { get; private set; }
    public Transform RightFoot { get; private set; }
    public Transform RightToes { get; private set; }

    private List<Material> Materials = new List<Material>();

    private InsoleManagerData Data;
    private bool IsGT;
    private bool RenderTrail;
    private float InitRootHeight;

    private quaternion[] OffsetRotations;

    public void SetData(InsoleManagerData data, bool isGT = false, bool renderTrail = true)
    {
        Data = data;
        IsGT = isGT;
        RenderTrail = renderTrail;
        InitSkeleton();
    }

    private void Start()
    {
        if (Data == null)
        {
            gameObject.SetActive(false);
            return;
        }
    }

    private void InitSkeleton()
    {
        // Create Skeleton
        BVHImporter importer = new BVHImporter();
        BVHAnimation tpose = importer.Import(GlobalManager.Instance.ExampleBVH, 1.0f, true);
        SkeletonTransforms = new Transform[tpose.Skeleton.Joints.Count];
        OffsetRotations = new quaternion[SkeletonTransforms.Length];
        for (int j = 0; j < SkeletonTransforms.Length; j++)
        {
            // Joints
            Skeleton.Joint joint = tpose.Skeleton.Joints[j];
            Transform t = (new GameObject()).transform;
            t.name = joint.Name;
            t.SetParent(j == 0 ? transform : SkeletonTransforms[joint.ParentIndex], false);
            t.localPosition = joint.LocalOffset;
            tpose.GetWorldPositionAndRotation(joint, 0, out quaternion worldRot, out _);
            t.rotation = worldRot;
            SkeletonTransforms[j] = t;
            // Visual
            Transform visual = (new GameObject()).transform;
            visual.name = "Visual";
            visual.SetParent(t, false);
            visual.localScale = new Vector3(0.1f, 0.1f, 0.1f);
            visual.localPosition = Vector3.zero;
            visual.localRotation = Quaternion.identity;
            // Sphere
            Transform sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
            sphere.name = "Sphere";
            sphere.SetParent(visual, false);
            sphere.localScale = Vector3.one;
            sphere.localPosition = Vector3.zero;
            sphere.localRotation = Quaternion.identity;
            Materials.Add(sphere.GetComponent<MeshRenderer>().material);
            Materials[^1].color = IsGT ? Color.green : Data.Color;
            Materials[^1].SetFloat("_Glossiness", Smoothness);
            // Capsule
            Transform capsule = GameObject.CreatePrimitive(PrimitiveType.Capsule).transform;
            capsule.name = "Capsule";
            capsule.SetParent(SkeletonTransforms[joint.ParentIndex].Find("Visual"), false);
            float distance = Vector3.Distance(t.position, t.parent.position) * (1.0f / visual.localScale.y) * 0.5f;
            capsule.localScale = new Vector3(0.5f, distance, 0.5f);
            Vector3 up = (t.position - t.parent.position).normalized;
            if (up.magnitude < 0.0001f)
            {
                continue;
            }
            capsule.localPosition = t.parent.InverseTransformDirection(up) * distance;
            capsule.localRotation = Quaternion.Inverse(t.parent.rotation) * Quaternion.LookRotation(new Vector3(-up.y, up.x, 0.0f), up);
            Materials.Add(capsule.GetComponent<MeshRenderer>().material);
            Materials[^1].color = IsGT ? Color.green : Data.Color;
            Materials[^1].SetFloat("_Glossiness", Smoothness);
            // References
            const float time = 5.0f;
            if (t.name == Data.LeftFoot)
            {
                LeftFoot = t;
                if (RenderTrail)
                {
                    Material trMat = new(Shader.Find("Particles/Standard Surface"));
                    trMat.SetOverrideTag("RenderType", "Transparent");
                    trMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                    trMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                    trMat.SetInt("_ZWrite", 0);
                    trMat.DisableKeyword("_ALPHATEST_ON");
                    trMat.DisableKeyword("_ALPHABLEND_ON");
                    trMat.EnableKeyword("_ALPHAPREMULTIPLY_ON");
                    trMat.renderQueue = 3000;
                    Materials.Add(trMat);
                    TrailRenderer tr = visual.gameObject.AddComponent<TrailRenderer>();
                    tr.startWidth = 0.03f;
                    tr.endWidth = 0.0f;
                    tr.time = time;
                    tr.material = trMat;
                    tr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    tr.minVertexDistance = 0.01f;
                    tr.numCornerVertices = 20;
                    tr.numCapVertices = 20;
                    Color c = IsGT ? Color.green : Data.Color;
                    tr.startColor = c;
                    c.a = 0.0f;
                    tr.endColor = c;
                }
            }
            else if (t.name == Data.LeftToes)
            {
                LeftToes = t;
            }
            else if (t.name == Data.RightFoot)
            {
                RightFoot = t;
                if (RenderTrail)
                {
                    Material trMat = new(Shader.Find("Particles/Standard Surface"));
                    trMat.SetOverrideTag("RenderType", "Transparent");
                    trMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                    trMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                    trMat.SetInt("_ZWrite", 0);
                    trMat.DisableKeyword("_ALPHATEST_ON");
                    trMat.DisableKeyword("_ALPHABLEND_ON");
                    trMat.EnableKeyword("_ALPHAPREMULTIPLY_ON");
                    trMat.renderQueue = 3000;
                    Materials.Add(trMat);
                    TrailRenderer tr = visual.gameObject.AddComponent<TrailRenderer>();
                    tr.startWidth = 0.03f;
                    tr.endWidth = 0.0f;
                    tr.time = time;
                    tr.material = Materials[^1];
                    tr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    tr.minVertexDistance = 0.01f;
                    tr.numCornerVertices = 20;
                    tr.numCapVertices = 20;
                    Color c = IsGT ? Color.green : Data.Color;
                    tr.startColor = c;
                    c.a = 0.0f;
                    tr.endColor = c;
                }
            }
            else if (t.name == Data.RightToes)
            {
                RightToes = t;
            }
            InitRootHeight = SkeletonTransforms[0].position.y;
        }
    }

    public void SetJoint(int index, Quaternion localRot, bool inertialize = false, bool smoothLegs = false, bool smoothRoot = false)
    {
        bool isLegs = 1 <= index && index <= 8;
        bool isRoot = 0 == index;
        float transitionSmoothness = isRoot || isLegs ? TransitionSmoothnessRoot : TransitionSmoothness;
        if (inertialize && (smoothLegs || !isLegs) && (smoothRoot || !isRoot))
        {
            float transitionDiff = index == 14 || index == 18 || index == 0 || index == 9 || index == 10 || index == 11 ? TransitionDiffSpine : TransitionDiff;
            if (math.dot(localRot, SkeletonTransforms[index].localRotation) < transitionDiff)
            {
                OffsetRotations[index] = math.mul(math.inverse(localRot), SkeletonTransforms[index].localRotation);
            }
            OffsetRotations[index] = math.slerp(OffsetRotations[index], quaternion.identity, math.clamp(transitionSmoothness * Time.deltaTime, 0.0f, 1.0f)); // decay
            localRot = math.mul(localRot, OffsetRotations[index]);
        }
        SkeletonTransforms[index].localRotation = localRot;
    }

    public void SetRootPosition(Vector3 pos)
    {
        SkeletonTransforms[0].position = pos;
    }

    private float AccumulatedRootHeightError;
    public void PreventFootUnderGround()
    {
        Vector3 root = SkeletonTransforms[0].position;
        root.y += AccumulatedRootHeightError;
        SkeletonTransforms[0].position = root;
        if (LeftToes.position.y < GlobalManager.Instance.Data[0].FloorHeight)
        {
            root = SkeletonTransforms[0].position;
            float diff = GlobalManager.Instance.Data[0].FloorHeight - LeftToes.position.y;
            AccumulatedRootHeightError += diff;
            root.y += diff;
            SkeletonTransforms[0].position = root;
        }
        if (RightToes.position.y < GlobalManager.Instance.Data[0].FloorHeight)
        {
            root = SkeletonTransforms[0].position;
            float diff = GlobalManager.Instance.Data[0].FloorHeight - RightToes.position.y;
            AccumulatedRootHeightError += diff;
            root.y += diff;
            SkeletonTransforms[0].position = root;
        }
    }

    private float PullVelocity;
    public void PullToGround()
    {
        Vector3 root = SkeletonTransforms[0].position;
        root.y += AccumulatedRootHeightError;
        SkeletonTransforms[0].position = root;
        float diff = InitRootHeight - root.y;
        diff = Mathf.SmoothDamp(diff, 0.0f, ref PullVelocity, 0.01f);
        root.y += diff;
        AccumulatedRootHeightError += diff;
        SkeletonTransforms[0].position = root;
    }

    private void OnDestroy()
    {
        if (Materials != null)
        {
            foreach (Material material in Materials)
            {
                Destroy(material);
            }
        }
    }
}
