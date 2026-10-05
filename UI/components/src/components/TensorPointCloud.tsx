import React, { useState } from 'react';

import { WebGLViewer } from './WebGLViewer';

type TensorPointCloudProps = {
  active: boolean;
};

const DATASET_OPTIONS = ['psi_final', 'A_final', 'N_a_stage', 'N_b_stage', 'N_c_stage'] as const;

export default function TensorPointCloud({ active }: TensorPointCloudProps) {
  const [configHash, setConfigHash] = useState('');
  const [datasetName, setDatasetName] = useState<(typeof DATASET_OPTIONS)[number]>('psi_final');

  return (
    <div className="space-y-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <h3 className="text-white font-bold mb-3">Tensor Point Cloud (WebGL)</h3>
        <div className="flex gap-2">
          <input
            value={configHash}
            onChange={(e) => setConfigHash(e.target.value)}
            placeholder="Enter config_hash"
            className="flex-1 bg-slate-950 border border-slate-700 rounded p-2 text-white text-sm font-mono"
          />
        </div>
        <div className="flex flex-wrap gap-2 mt-3">
          {DATASET_OPTIONS.map((option) => {
            const selected = option === datasetName;
            return (
              <button
                key={option}
                type="button"
                onClick={() => setDatasetName(option)}
                className={selected
                  ? 'px-3 py-1 rounded-full text-xs font-mono bg-cyan-400 text-slate-950'
                  : 'px-3 py-1 rounded-full text-xs font-mono bg-slate-950 border border-slate-700 text-slate-300'}
              >
                {option}
              </button>
            );
          })}
        </div>
      </div>

      {active ? <WebGLViewer configHash={configHash.trim()} datasetName={datasetName} /> : null}
    </div>
  );
}
