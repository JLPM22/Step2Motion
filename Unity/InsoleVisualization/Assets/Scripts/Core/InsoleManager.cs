using BVH;
using System.IO;
using System.Runtime.Serialization.Formatters;
using Unity.Mathematics;
using UnityEngine;

public class InsoleManager : MonoBehaviour
{
    public int Frame = 0;
    public bool Stop = false;
    public bool Restart = false;
    public bool SetFramerate = true;
    public bool CopyRoot = false;
    public bool ResetRoot = false;
    public float Separation = 0.5f;
    public bool SmoothTransitions = false;
    public bool SmoothRoot = false;
    public bool SmoothLegs = false;
    public bool ShowTotalForce = true;

    [Header("Post-processing")]
    public bool PreventFootUnderGround = false;
    public bool PullToGround = false;

    [Header("Skeleton")]
    public float Scale = 1.0f;
    public bool ShowGT = true;
    public bool[] ShowPred;

    [Header("Insole")]
    public GameObject LeftInsolePrefab;
    public GameObject RightInsolePrefab;
    public Color MaxForceColor = Color.red;

    private InsoleManagerData[] Data;
    private SkeletonAvatar[] Avatar;
    private BVHAnimation GTAnimation;
    private BVHAnimation[] PredAnimation;
    private InsoleFeatures[] InsoleFeatures;

    public InsoleFeatures.Features[] CurrentFeatures { get; private set; }
    private int MaxFrames;
    private int TargetFrameRate;
    private float TargetFrameRateFactor;
    private Color BasePressureColor = new Color(0.3f, 0.3f, 0.3f, 1.0f);
    private GameObject LeftInsoleGT;
    private GameObject RightInsoleGT;
    private Material[] LeftInsoleGTMats;
    private Material[] RightInsoleGTMats;
    private GameObject[] LeftInsolePred;
    private GameObject[] RightInsolePred;
    private Material[][] LeftInsolePredMats;
    private Material[][] RightInsolePredMats;
    private MeshRenderer[] LeftVisualGT;
    private MeshRenderer[] RightVisualGT;
    private MeshRenderer[][] LeftVisualPred;
    private MeshRenderer[][] RightVisualPred;
    private bool IsInsoleShown = false;
    private bool IsForceShown = false; // otherwise acceleration is shown
    private Vector3[] RootMotionCorrection;

    private SkeletonAvatar SkeletonGT
    {
        get => GlobalManager.Instance.SkeletonGT;
    }

    private void Awake()
    {
        Data = GlobalManager.Instance.Data;
        // Load ground truth animation
        string predictionName = GlobalManager.Instance.PredictionName;
        string dirPath = Path.Combine(Data[0].ModelsPath, Data[0].ModelName, "predictions");
        string pathGT = Path.Combine(dirPath, predictionName + "_gt.bvh");
        string bvhPathGT = File.ReadAllText(pathGT);
        BVHImporter gtImporter = new();
        GTAnimation = gtImporter.Import(bvhPathGT, Scale);
        // Set target framerate
        MaxFrames = GTAnimation.Frames.Length;
        if (SetFramerate)
        {
            TargetFrameRate = Mathf.RoundToInt(1.0f / GTAnimation.FrameTime);
            TargetFrameRateFactor = 1;
            Application.targetFrameRate = TargetFrameRate;
        }
        if (Data.Length == 1 && Data[0].Seeds != null && Data[0].Seeds.Length > 1)
        {
            // multiple seeds mode
            int numberOfSeeds = Data[0].Seeds.Length;
            PredAnimation = new BVHAnimation[numberOfSeeds];
            InsoleFeatures = new InsoleFeatures[numberOfSeeds];
            RootMotionCorrection = new Vector3[numberOfSeeds];
            CurrentFeatures = new InsoleFeatures.Features[numberOfSeeds];
            for (int i = 0; i < numberOfSeeds; ++i)
            {
                string pathPred = Path.Combine(dirPath, predictionName + "_pred_s" + Data[0].Seeds[i].ToString() + ".bvh");
                string bvhPathPred = File.ReadAllText(pathPred);
                BVHImporter predImporter = new();
                PredAnimation[i] = predImporter.Import(bvhPathPred, Scale);
                // Insole Visualization
                string pathInsole = Path.Combine(dirPath, predictionName + "_cs.json");
                InsoleFeatures[i] = new InsoleFeatures();
                InsoleFeatures[i].ImportFromJSON(pathInsole);
            }
            if (ShowPred == null || ShowPred.Length != numberOfSeeds)
            {
                ShowPred = new bool[numberOfSeeds];
                for (int i = 0; i < ShowPred.Length; ++i)
                {
                    ShowPred[i] = true;
                }
            }
        }
        else
        {
            // normal comparison mode
            PredAnimation = new BVHAnimation[Data.Length];
            InsoleFeatures = new InsoleFeatures[Data.Length];
            RootMotionCorrection = new Vector3[Data.Length];
            CurrentFeatures = new InsoleFeatures.Features[Data.Length];
            for (int i = 0; i < Data.Length; ++i)
            {
                dirPath = Path.Combine(Data[i].ModelsPath, Data[i].ModelName, "predictions");
                string pathPred = Path.Combine(dirPath, predictionName + "_pred.bvh");
                string bvhPathPred = File.ReadAllText(pathPred);
                BVHImporter predImporter = new();
                PredAnimation[i] = predImporter.Import(bvhPathPred, Scale);
                // Insole Visualization
                string pathInsole = Path.Combine(dirPath, predictionName + "_cs.json");
                InsoleFeatures[i] = new InsoleFeatures();
                InsoleFeatures[i].ImportFromJSON(pathInsole);
            }
            if (ShowPred == null || ShowPred.Length != Data.Length)
            {
                ShowPred = new bool[Data.Length];
                for (int i = 0; i < ShowPred.Length; ++i)
                {
                    ShowPred[i] = true;
                }
            }
        }
    }

    private void Start()
    {
        Avatar = GlobalManager.Instance.Avatars;
    }

    private void Update()
    {
        if (LeftVisualGT == null)
        {
            InitVisuals();
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            Stop = !Stop;
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            Restart = true;
        }

        if (Input.GetKeyDown(KeyCode.I))
        {
            IsInsoleShown = !IsInsoleShown;
        }
        if (Input.GetKeyDown(KeyCode.F))
        {
            IsForceShown = !IsForceShown;
        }

        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            Frame += 1;
        }
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            Frame -= 1;
        }
        if (Input.GetKeyDown(KeyCode.UpArrow))
        {
            TargetFrameRateFactor = Mathf.Clamp(TargetFrameRateFactor * 2, 1.0f / 8.0f, 4.0f);
            Application.targetFrameRate = Mathf.RoundToInt(TargetFrameRate * TargetFrameRateFactor);
        }
        if (Input.GetKeyDown(KeyCode.DownArrow))
        {
            TargetFrameRateFactor = Mathf.Clamp(TargetFrameRateFactor / 2, 1.0f / 8.0f, 4.0f);
            Application.targetFrameRate = Mathf.RoundToInt(TargetFrameRate * TargetFrameRateFactor);
        }
        Frame = Mathf.Clamp(Frame, 0, MaxFrames);

        BVHAnimation.Frame gtFrame = GTAnimation.Frames[Frame];
        SkeletonGT.SetRootPosition(gtFrame.RootMotion);

        for (int j = 0; j < gtFrame.LocalRotations.Length; ++j)
        {
            SkeletonGT.SetJoint(j, gtFrame.LocalRotations[j]);
        }

        SkeletonGT.gameObject.SetActive(ShowGT);

        CurrentFeatures[0] = InsoleFeatures[0].GetFrame(Frame);
        UpdateTotalForce(LeftVisualGT, ShowTotalForce ? CurrentFeatures[0].LeftTotalForce : 0.0f, Color.green, 0);
        UpdateTotalForce(RightVisualGT, ShowTotalForce ? CurrentFeatures[0].RightTotalForce : 0.0f, Color.green, 0);
        UpdatePressure(LeftInsoleGT.transform, LeftInsoleGTMats, CurrentFeatures[0].LeftPressure, 0);
        UpdatePressure(RightInsoleGT.transform, RightInsoleGTMats, CurrentFeatures[0].RightPressure, 0);

        if (IsInsoleShown)
        {
            LeftInsoleGT.SetActive(true);
            RightInsoleGT.SetActive(true);
            foreach (MeshRenderer mr in LeftVisualGT) mr.gameObject.SetActive(false);
            foreach (MeshRenderer mr in RightVisualGT) mr.gameObject.SetActive(false);
        }
        else
        {
            LeftInsoleGT.SetActive(false);
            RightInsoleGT.SetActive(false);
            foreach (MeshRenderer mr in LeftVisualGT) mr.gameObject.SetActive(true);
            foreach (MeshRenderer mr in RightVisualGT) mr.gameObject.SetActive(true);
        }

        for (int i = 0; i < PredAnimation.Length; ++i)
        {
            BVHAnimation.Frame predFrame = PredAnimation[i].Frames[Frame];

            if (ResetRoot)
            {
                RootMotionCorrection[i] = gtFrame.RootMotion - predFrame.RootMotion;
            }
            Vector3 rootPosition = CopyRoot ? gtFrame.RootMotion : predFrame.RootMotion + RootMotionCorrection[i];
            rootPosition += Vector3.right * Separation * (i + 1);
            Avatar[i].SetRootPosition(rootPosition);

            for (int j = 0; j < gtFrame.LocalRotations.Length; ++j)
            {
                Avatar[i].SetJoint(j, predFrame.LocalRotations[j], inertialize: SmoothTransitions, smoothLegs: SmoothLegs, smoothRoot: SmoothRoot);
            }

            if (PreventFootUnderGround)
            {
                Avatar[i].PreventFootUnderGround();
            }
            if (PullToGround)
            {
                Avatar[i].PullToGround();
            }

            Avatar[i].gameObject.SetActive(ShowPred[i]);

            CurrentFeatures[i] = InsoleFeatures[i].GetFrame(Frame);
            Color color = Data[0].Color;
            if (Data.Length == PredAnimation.Length)
            {
                color = Data[i].Color;
            }
            UpdateTotalForce(LeftVisualPred[i], ShowTotalForce ? CurrentFeatures[i].LeftTotalForce : 0.0f, color, i);
            UpdateTotalForce(RightVisualPred[i], ShowTotalForce ? CurrentFeatures[i].RightTotalForce : 0.0f, color, i);
            UpdatePressure(LeftInsolePred[i].transform, LeftInsolePredMats[i], CurrentFeatures[i].LeftPressure, i);
            UpdatePressure(RightInsolePred[i].transform, RightInsolePredMats[i], CurrentFeatures[i].RightPressure, i);

            if (IsInsoleShown)
            {
                LeftInsolePred[i].SetActive(true);
                RightInsolePred[i].SetActive(true);
                foreach (MeshRenderer mr in LeftVisualPred[i]) mr.gameObject.SetActive(false);
                foreach (MeshRenderer mr in RightVisualPred[i]) mr.gameObject.SetActive(false);
            }
            else
            {
                LeftInsolePred[i].SetActive(false);
                RightInsolePred[i].SetActive(false);
                foreach (MeshRenderer mr in LeftVisualPred[i]) mr.gameObject.SetActive(true);
                foreach (MeshRenderer mr in RightVisualPred[i]) mr.gameObject.SetActive(true);
            }
        }

        if (ResetRoot)
        {
            ResetRoot = false;
        }

        if (!Stop) Frame = (Frame + 1) % MaxFrames;
        if (Restart)
        {
            Frame = 0;
            Restart = false;
        }
    }

    private void InitVisuals()
    {
        LeftVisualGT = SkeletonGT.LeftFoot.GetComponentsInChildren<MeshRenderer>();
        RightVisualGT = SkeletonGT.RightFoot.GetComponentsInChildren<MeshRenderer>();
        LeftVisualPred = new MeshRenderer[Avatar.Length][];
        RightVisualPred = new MeshRenderer[Avatar.Length][];
        for (int i = 0; i < Avatar.Length; ++i)
        {
            LeftVisualPred[i] = Avatar[i].LeftFoot.GetComponentsInChildren<MeshRenderer>();
            RightVisualPred[i] = Avatar[i].RightFoot.GetComponentsInChildren<MeshRenderer>();
        }

        LeftInsoleGT = Instantiate(LeftInsolePrefab, SkeletonGT.LeftFoot);
        LeftInsoleGT.transform.SetLocalPositionAndRotation(Data[0].LeftLocalPosition, Data[0].LeftLocalRotation);
        MeshRenderer[] mrsGT = LeftInsoleGT.transform.GetComponentsInChildren<MeshRenderer>();
        LeftInsoleGTMats = new Material[mrsGT.Length];
        for (int i = 0; i < mrsGT.Length; ++i)
        {
            LeftInsoleGTMats[i] = mrsGT[i].material;
        }
        RightInsoleGT = Instantiate(RightInsolePrefab, SkeletonGT.RightFoot);
        RightInsoleGT.transform.SetLocalPositionAndRotation(Data[0].RightLocalPosition, Data[0].RightLocalRotation);
        mrsGT = RightInsoleGT.transform.GetComponentsInChildren<MeshRenderer>();
        RightInsoleGTMats = new Material[mrsGT.Length];
        for (int i = 0; i < mrsGT.Length; ++i)
        {
            RightInsoleGTMats[i] = mrsGT[i].material;
        }
        LeftInsolePred = new GameObject[Avatar.Length];
        RightInsolePred = new GameObject[Avatar.Length];
        LeftInsolePredMats = new Material[Avatar.Length][];
        RightInsolePredMats = new Material[Avatar.Length][];
        for (int i = 0; i < Avatar.Length; ++i)
        {
            int indexData = Data.Length == Avatar.Length ? i : 0;
            LeftInsolePred[i] = Instantiate(LeftInsolePrefab, Avatar[i].LeftFoot);
            LeftInsolePred[i].transform.SetLocalPositionAndRotation(Data[indexData].LeftLocalPosition, Data[indexData].LeftLocalRotation);
            MeshRenderer[] mrs = LeftInsolePred[i].transform.GetComponentsInChildren<MeshRenderer>();
            LeftInsolePredMats[i] = new Material[mrs.Length];
            for (int j = 0; j < mrs.Length; ++j)
            {
                LeftInsolePredMats[i][j] = mrs[j].material;
            }
            RightInsolePred[i] = Instantiate(RightInsolePrefab, Avatar[i].RightFoot);
            RightInsolePred[i].transform.SetLocalPositionAndRotation(Data[indexData].RightLocalPosition, Data[indexData].RightLocalRotation);
            mrs = RightInsolePred[i].transform.GetComponentsInChildren<MeshRenderer>();
            RightInsolePredMats[i] = new Material[mrs.Length];
            for (int j = 0; j < mrs.Length; ++j)
            {
                RightInsolePredMats[i][j] = mrs[j].material;
            }
        }
    }

    private void UpdatePressure(Transform insole, Material[] mats, float[] pressure, int avatarIndex)
    {
        for (int i = 0; i < pressure.Length; i++)
        {
            mats[i].color = Color.Lerp(BasePressureColor, MaxForceColor, Mathf.Clamp01(pressure[i] / InsoleFeatures[avatarIndex].MaxTotalPressure));
            insole.GetChild(i).localScale = new Vector3(1.0f, 1.0f, Mathf.Clamp01(pressure[i] / InsoleFeatures[avatarIndex].MaxTotalPressure) * 10 + 1);
        }
    }

    private void UpdateTotalForce(MeshRenderer[] mrs, float totalForce, Color baseColor, int avatarIndex)
    {
        foreach (MeshRenderer mr in mrs)
        {
            mr.sharedMaterial.color = Color.Lerp(baseColor, MaxForceColor, Mathf.Clamp01(totalForce / InsoleFeatures[avatarIndex].MaxTotalForce));
        }
    }

    private void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        void DrawForce(float3 origin, float3 force, int avatarIndex)
        {
            Gizmos.color = Color.red;
            float mag = math.length(force);
            GizmosExtensions.DrawArrow(origin, origin + force * 0.0005f, arrowHeadLength: mag * 0.00005f, thickness: 10 * (mag / InsoleFeatures[avatarIndex].MaxTotalForce));
        }

        void DrawAcceleration(float3 origin, float3 acc, int avatarIndex)
        {
            Gizmos.color = Color.black;
            float accMag = math.length(acc);
            GizmosExtensions.DrawArrow(origin, origin + acc * 0.25f, arrowHeadLength: accMag * 0.1f, thickness: 10 * (accMag / InsoleFeatures[avatarIndex].MaxOneComponentAcceleration));
        }

        if (IsForceShown)
        {
            float3 leftGRF = math.normalize(CurrentFeatures[0].LeftAcceleration) * CurrentFeatures[0].LeftTotalForce;
            float3 rightGRF = math.normalize(CurrentFeatures[0].RightAcceleration) * CurrentFeatures[0].RightTotalForce;

            if (ShowGT)
            {
                DrawForce((SkeletonGT.LeftFoot.position + SkeletonGT.LeftToes.position) / 2.0f, leftGRF, 0);
                DrawForce((SkeletonGT.RightFoot.position + SkeletonGT.RightToes.position) / 2.0f, rightGRF, 0);
            }
        }
        else
        {
            float3 leftAcc = CurrentFeatures[0].LeftAcceleration;
            float3 rightAcc = CurrentFeatures[0].RightAcceleration;

            if (ShowGT)
            {
                DrawAcceleration((SkeletonGT.LeftFoot.position + SkeletonGT.LeftToes.position) / 2.0f, leftAcc, 0);
                DrawAcceleration((SkeletonGT.RightFoot.position + SkeletonGT.RightToes.position) / 2.0f, rightAcc, 0);
            }
        }

        for (int i = 0; i < Data.Length; ++i)
        {
            if (IsForceShown)
            {
                float3 leftGRF = math.normalize(CurrentFeatures[i].LeftAcceleration) * CurrentFeatures[i].LeftTotalForce;
                float3 rightGRF = math.normalize(CurrentFeatures[i].RightAcceleration) * CurrentFeatures[i].RightTotalForce;

                if (ShowPred[i])
                {
                    DrawForce((Avatar[i].LeftFoot.position + Avatar[i].LeftToes.position) / 2.0f, leftGRF, i);
                    DrawForce((Avatar[i].RightFoot.position + Avatar[i].RightToes.position) / 2.0f, rightGRF, i);
                }
            }
            else
            {
                float3 leftAcc = CurrentFeatures[i].LeftAcceleration;
                float3 rightAcc = CurrentFeatures[i].RightAcceleration;

                if (ShowPred[i])
                {
                    DrawAcceleration((Avatar[i].LeftFoot.position + Avatar[i].LeftToes.position) / 2.0f, leftAcc, i);
                    DrawAcceleration((Avatar[i].RightFoot.position + Avatar[i].RightToes.position) / 2.0f, rightAcc, i);
                }
            }
        }
    }

    private void OnDestroy()
    {
        if (LeftInsoleGTMats != null)
        {
            foreach (Material mat in LeftInsoleGTMats)
            {
                Destroy(mat);
            }
        }
        if (RightInsoleGTMats != null)
        {
            foreach (Material mat in RightInsoleGTMats)
            {
                Destroy(mat);
            }
        }
        if (LeftInsolePredMats != null)
        {
            for (int i = 0; i < LeftInsolePredMats.Length; ++i)
            {
                foreach (Material mat in LeftInsolePredMats[i])
                {
                    Destroy(mat);
                }
            }
        }
        if (RightInsolePredMats != null)
        {
            for (int i = 0; i < RightInsolePredMats.Length; ++i)
            {
                foreach (Material mat in RightInsolePredMats[i])
                {
                    Destroy(mat);
                }
            }
        }
    }
}
