import React, { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';

import { getPointCloudBuffer } from '../api_client';

interface WebGLViewerProps {
  configHash: string;
  datasetName?: string;
}

type GeometryData = {
  positions: Float32Array;
  colors: Float32Array;
};

const PointCloud = ({ geometryData }: { geometryData: GeometryData }) => {
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={geometryData.positions.length / 3} array={geometryData.positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={geometryData.colors.length / 3} array={geometryData.colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.3}
        vertexColors
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
};

export const WebGLViewer = ({ configHash, datasetName = 'psi_final' }: WebGLViewerProps) => {
  const [geometryData, setGeometryData] = useState<GeometryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!configHash) {
      setGeometryData(null);
      setLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;

    const fetchTensor = async () => {
      setLoading(true);
      setError(null);
      try {
        const buffer = await getPointCloudBuffer(configHash, datasetName, 0.05);
        const rawData = new Float32Array(buffer);

        const numPoints = rawData.length / 4;
        const posArray = new Float32Array(numPoints * 3);
        const colArray = new Float32Array(numPoints * 3);

        const colorMap = new THREE.Color();

        for (let i = 0; i < numPoints; i++) {
          posArray[i * 3] = rawData[i * 4];
          posArray[i * 3 + 1] = rawData[i * 4 + 1];
          posArray[i * 3 + 2] = rawData[i * 4 + 2];

          const intensity = rawData[i * 4 + 3];
          colorMap.setHSL(0.55 - (intensity * 0.4), 1.0, 0.3 + (intensity * 0.5));
          colArray[i * 3] = colorMap.r;
          colArray[i * 3 + 1] = colorMap.g;
          colArray[i * 3 + 2] = colorMap.b;
        }

        if (!cancelled) {
          setGeometryData({ positions: posArray, colors: colArray });
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to fetch tensor data');
          setGeometryData(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void fetchTensor();

    return () => {
      cancelled = true;
    };
  }, [configHash, datasetName]);

  return (
    <div style={{ width: '100%', height: '500px', backgroundColor: '#050505', borderRadius: '8px', border: '1px solid #333', position: 'relative' }}>
      <Canvas camera={{ position: [25, 25, 25], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <Stars radius={50} depth={50} count={2000} factor={4} saturation={0} fade speed={1} />
        {geometryData ? <PointCloud geometryData={geometryData} /> : null}
        <OrbitControls autoRotate autoRotateSpeed={1.0} enablePan={true} enableZoom={true} />
      </Canvas>
      <div style={{ position: 'absolute', top: 10, left: 10, color: '#00ffcc', fontFamily: 'monospace', fontSize: '12px' }}>
        LIVE TENSOR: {datasetName}
      </div>
      {loading ? (
        <div style={{ position: 'absolute', top: 32, left: 10, color: '#8ef5db', fontFamily: 'monospace', fontSize: '12px' }}>
          Loading point cloud...
        </div>
      ) : null}
      {error ? (
        <div style={{ position: 'absolute', top: 32, left: 10, color: '#ff8a8a', fontFamily: 'monospace', fontSize: '12px' }}>
          {error}
        </div>
      ) : null}
    </div>
  );
};
