using Cinemachine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraManager : MonoBehaviour
{
    public float SmoothFactor = 1.0f;

    private CinemachineFreeLook CinemachineFreeLook;

    private SkeletonAvatar SkeletonGT;
    private SkeletonAvatar[] SkeletonPred;
    private Transform TargetGT;
    private Transform TargetPred;
    private Transform TargetScene;

    private float TopRadius;
    private float MidRadius;
    private float BotRadius;

    private int CurrentPredIndex = 0;

    private void Awake()
    {
        CinemachineFreeLook = GetComponentInChildren<CinemachineFreeLook>();
        TargetGT = new GameObject("TargetGT").transform;
        TargetGT.SetParent(transform, true);
        TargetPred = new GameObject("TargetPred").transform;
        TargetPred.SetParent(transform, true);
        TargetScene = new GameObject("TargetScene").transform;
        TargetScene.SetParent(transform, true);
        TopRadius = CinemachineFreeLook.m_Orbits[0].m_Radius;
        MidRadius = CinemachineFreeLook.m_Orbits[1].m_Radius;
        BotRadius = CinemachineFreeLook.m_Orbits[2].m_Radius;
    }

    private void Start()
    {
        SkeletonGT = GlobalManager.Instance.SkeletonGT;
        SkeletonPred = GlobalManager.Instance.Avatars;
        SetSceneMode();
    }

    private void LateUpdate()
    {
        if (Input.GetKeyDown(KeyCode.G))
        {
            CinemachineFreeLook.Follow = TargetGT;
            CinemachineFreeLook.LookAt = TargetGT;
            CinemachineFreeLook.m_Orbits[0].m_Radius = TopRadius;
            CinemachineFreeLook.m_Orbits[1].m_Radius = MidRadius;
            CinemachineFreeLook.m_Orbits[2].m_Radius = BotRadius;
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            SetSceneMode();
        }
        if (Input.GetKeyDown(KeyCode.P))
        {
            CinemachineFreeLook.Follow = TargetPred;
            CinemachineFreeLook.LookAt = TargetPred;
            CinemachineFreeLook.m_Orbits[0].m_Radius = TopRadius;
            CinemachineFreeLook.m_Orbits[1].m_Radius = MidRadius;
            CinemachineFreeLook.m_Orbits[2].m_Radius = BotRadius;

            CurrentPredIndex = (CurrentPredIndex + 1) % SkeletonPred.Length;
        }

        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
        {
            CinemachineFreeLook.m_XAxis.m_InputAxisValue = Input.GetAxis("Mouse X");
        }

        TargetGT.position = Vector3.Lerp(TargetGT.position, (SkeletonGT.LeftToes.position + SkeletonGT.RightToes.position) / 2.0f, Time.deltaTime * SmoothFactor);
        TargetPred.position = Vector3.Lerp(TargetPred.position, (SkeletonPred[CurrentPredIndex].LeftToes.position + SkeletonPred[CurrentPredIndex].RightToes.position) / 2.0f + Vector3.up * 0.5f, Time.deltaTime * SmoothFactor);
    }

    private void SetSceneMode()
    {
        CinemachineFreeLook.Follow = TargetScene;
        CinemachineFreeLook.LookAt = TargetScene;
        CinemachineFreeLook.m_Orbits[0].m_Radius = TopRadius * 2;
        CinemachineFreeLook.m_Orbits[1].m_Radius = MidRadius * 2;
        CinemachineFreeLook.m_Orbits[2].m_Radius = BotRadius * 2;
    }
}
