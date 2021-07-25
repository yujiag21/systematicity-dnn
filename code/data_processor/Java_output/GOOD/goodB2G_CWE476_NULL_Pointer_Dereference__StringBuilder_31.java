    private void goodB2G() throws Throwable
    {
        StringBuilder dataCopy;
        {
            StringBuilder data;

            /* POTENTIAL FLAW: data is null */
            data = null;

            dataCopy = data;
        }
        {
            StringBuilder data = dataCopy;

            /* FIX: validate that data is non-null */
            if (data != null)
            {
                IO.writeLine("" + data.length());
            }
            else
            {
                IO.writeLine("data is null");
            }

        }
    }
