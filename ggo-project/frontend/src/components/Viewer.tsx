import type { ReactElement } from 'react';

export type ViewerProps = {
  title?: string;
};

function Viewer({ title = 'CT Viewer' }: ViewerProps): ReactElement {
  return <section>{title}</section>;
}

export default Viewer;
