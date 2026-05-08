import type { ReactElement } from 'react';

import Controls from '../components/Controls';
import Upload from '../components/Upload';
import Viewer from '../components/Viewer';

function Home(): ReactElement {
  return (
    <main>
      <h1>GGO Project</h1>
      <Upload />
      <Controls />
      <Viewer />
    </main>
  );
}

export default Home;
