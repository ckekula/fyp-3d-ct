import type { ReactElement } from 'react';

export type ControlsProps = {
  onRun?: () => void;
};

function Controls({ onRun }: ControlsProps): ReactElement {
  return (
    <button type="button" onClick={onRun}>
      Run Analysis
    </button>
  );
}

export default Controls;
